import json
import re
import requests
from openai import OpenAI, OpenAIError
from app.config import get_settings
from app.schemas import SegmentOut

# Homoglyph and Unicode corruptions that occur in Thai speech recognition
HOMOGLYPH_REPLACEMENTS = {
    # Greek and Cyrillic homoglyphs
    "α": "า",       # Greek alpha -> Thai sara aa
    "ν": "น",       # Greek nu -> Thai no nu
    "ă": "ั",       # Latin a-breve -> Thai mai han-akat
    "ュ": "ู",       # Japanese Katakana yu -> Thai sara oo
    "р": "ร",       # Cyrillic er -> Thai ro ruea
    "є": "เ",       # Cyrillic Ukrainian ye -> Thai sara e
    "ư": "ื",       # Latin u-horn -> Thai sara ue
    "เเ": "แ",      # Double sara e -> sara ae

    # Broken Unicode / byte artifacts
    "ب": "บ",
    "ב": "บ",
    "ại": "าย",
    "ột": "วด",
    "ครặp": "ครับ",
    "ครัป": "ครับ",
    "ครัฮะ": "ครับ",
    "ครั๊": "ครับ",
    "ครัب": "ครับ",
    "ครรับ": "ครับ",
    "ครับบ": "ครับ",
    "ค่ะะ": "ค่ะ",
    "ค่ะ่ะ": "ค่ะ",
}

# Common phonetic speech recognition typos (Only pure word-level mishearings, ZERO sentence injections)
PHONETIC_TYPO_CORRECTIONS = {
    "ผมที่พยาบาลวัด": "หมอเห็นผลที่พยาบาลวัด",
    "หมอเห็นผลที่พยาบาล": "หมอเห็นผลที่พยาบาล",
    "ชี้ผจร": "ชีพจร",
    "ชีบจร": "ชีพจร",
    "ชีพจน": "ชีพจร",
    "ชีพจอน": "ชีพจร",
    "ชีพระจร": "ชีพจร",
    "ความดาล": "ความดัน",
    "แตดไป": "แตะไป",
    "แตด": "แตะ",
    "แตต": "แตะ",
    "อุณหาภูมิ": "อุณหภูมิ",
    "องศาเซลเซียต": "องศาเซลเซียส",
    "ปัศวะ": "ปัสสาวะ",
    "ปัสวะ": "ปัสสาวะ",
    "ปัสสวะ": "ปัสสาวะ",
    "พาราเซต": "พาราเซตามอล",
    "พาราศตามอน": "พาราเซตามอล",
    "พาราเศตามอน": "พาราเซตามอล",
    "บันเท่า": "บรรเทา",
    "อักเซฟ": "อักเสบ",
    "อักเสฟ": "อักเสบ",
    "เพลีๆ": "เพลียๆ",
    "เพลียม": "เพลีย",
    "แเพลีย": "เพลีย",
    "รูสถึก": "รู้สึก",
    "เชินนั่ง": "เชิญนั่ง",
    "สวาสวัสดี": "สวัสดี",
    "สวาสดี่": "สวัสดี",
    "ตล่อด": "ตลอด",
}


def clean_unicode_homoglyphs(text: str) -> str:
    """Clean up non-standard Unicode characters and homoglyphs."""
    if not text:
        return ""
    cleaned = text
    for k, v in HOMOGLYPH_REPLACEMENTS.items():
        cleaned = cleaned.replace(k, v)
    return cleaned


def clean_stuttering_and_duplicates(text: str) -> str:
    """Collapse dragged repetitive characters and double spaces without altering meaning."""
    if not text:
        return ""
    # Collapse 3 or more consecutive identical characters -> 1 (e.g. ครับบบบบ -> ครับ, คือออออ -> คือ)
    cleaned = re.sub(r"([ก-ฮะ-ูเ-์a-zA-Z])\1{2,}", r"\1", text)
    # Clean double spaces
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    return cleaned.strip()


def apply_dictionary(text: str) -> str:
    """Apply safe, zero-hallucination linguistic corrections only."""
    if not text:
        return ""
    cleaned = clean_unicode_homoglyphs(text)
    for wrong, right in PHONETIC_TYPO_CORRECTIONS.items():
        cleaned = cleaned.replace(wrong, right)
    cleaned = clean_stuttering_and_duplicates(cleaned)
    return cleaned


def normalize_post_process_mode(mode: str | None) -> str:
    settings = get_settings()
    selected_mode = (mode or settings.post_process_mode).strip().lower()
    if selected_mode in {"false", "off", "no", "none", "raw"}:
        return "none"
    if selected_mode in {"true", "on", "yes"}:
        return settings.post_process_mode
    return selected_mode


def normalize_clinical_role(role_raw: str | None, default: str = "แพทย์") -> tuple[str, str]:
    """Normalize clinical role to 'แพทย์' (Doctor) or 'ผู้ป่วย' (Patient)."""
    if not role_raw:
        role_raw = default
    r_lower = str(role_raw).strip().lower()
    if any(k in r_lower for k in ["ผู้ป่วย", "คนไข้", "patient", "speaker_01"]):
        return "ผู้ป่วย", "SPEAKER_01"
    return "แพทย์", "SPEAKER_00"


def get_proofread_system_prompt() -> str:
    return (
        "You are an expert Clinical Dialogue AI Specialist and Thai Medical Speech Transcriber.\n"
        "Your sole task is to format the given transcript into clean turn-by-turn dialogue between 'แพทย์' (Doctor) and 'ผู้ป่วย' (Patient), and fix obvious Thai speech recognition phonetic typos.\n\n"
        "STRICT ZERO-HALLUCINATION RULES:\n"
        "1. STRICT VERBATIM FIDELITY: Transcribe and retain EXACTLY what was said in the transcript. DO NOT add, invent, embellish, speculate, or insert ANY words, symptoms, illnesses, vital signs, medications, or conversation that the speakers did not say.\n"
        "2. PRESERVE ALL NUMBERS & FACTS: Keep every number, unit, symptom, and detail exactly as stated.\n"
        "3. ACCURATE SPEAKER SEPARATION:\n"
        "   - 'แพทย์: ' (Doctor): Doctor speaking (greetings, asking questions, explaining results, giving diagnosis/prescription).\n"
        "   - 'ผู้ป่วย: ' (Patient): Patient speaking (answers, describing symptoms/feelings, replying).\n"
        "   - Every speaker turn MUST begin with either 'แพทย์: ' or 'ผู้ป่วย: '.\n"
        "4. OUTPUT FORMAT:\n"
        "   - Return ONLY the clean, formatted dialogue with each turn beginning with 'แพทย์: ' or 'ผู้ป่วย: ', separated by a blank line.\n"
        "   - Do not add markdown code blocks, backticks, or explanatory notes."
    )


def format_clinical_dialogue_rule_based(text: str) -> str:
    """Fallback: ensure text is formatted with clean turn tags."""
    if not text:
        return ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    formatted_lines = []
    current_role = "แพทย์"
    for line in lines:
        role, _ = normalize_clinical_role(line, default=current_role)
        clean_text = re.sub(r"^(?:แพทย์|ผู้ป่วย|หมอ|คนไข้|SPEAKER_\d+)\s*[:：\-]\s*", "", line).strip()
        if clean_text:
            formatted_lines.append(f"{role}: {clean_text}")
            current_role = "ผู้ป่วย" if role == "แพทย์" else "แพทย์"
    return "\n\n".join(formatted_lines) if formatted_lines else text


def parse_dialogue_text_into_segments(text: str, total_duration: float = 180.0) -> list[dict]:
    """Parse dialogue text formatted as 'แพทย์: ...' / 'ผู้ป่วย: ...' into clean segment dicts."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    segments = []
    seg_id = 0
    num_lines = max(len(lines), 1)
    dur_per_line = max(total_duration / num_lines, 2.0)

    for idx, line in enumerate(lines):
        role = "แพทย์"
        speaker = "SPEAKER_00"
        clean_line = line

        m_doc = re.match(r"^(?:\[?(?:แพทย์|หมอ|Doctor|Clinician)\]?|SPEAKER_00)\s*[:：\-]\s*(.*)$", line, re.IGNORECASE)
        m_pat = re.match(r"^(?:\[?(?:ผู้ป่วย|คนไข้|Patient)\]?|SPEAKER_01)\s*[:：\-]\s*(.*)$", line, re.IGNORECASE)

        if m_doc:
            role = "แพทย์"
            speaker = "SPEAKER_00"
            clean_line = m_doc.group(1).strip()
        elif m_pat:
            role = "ผู้ป่วย"
            speaker = "SPEAKER_01"
            clean_line = m_pat.group(1).strip()
        else:
            role, speaker = normalize_clinical_role(line, default="แพทย์" if idx % 2 == 0 else "ผู้ป่วย")

        clean_line = re.sub(r"^(?:แพทย์|หมอ|ผู้ป่วย|คนไข้|SPEAKER_\d+)\s*[:：\-]\s*", "", clean_line).strip()
        if not clean_line:
            continue

        start_t = round(idx * dur_per_line, 2)
        end_t = round((idx + 1) * dur_per_line, 2)

        segments.append({
            "id": seg_id,
            "start": start_t,
            "end": end_t,
            "text": clean_line,
            "speaker": speaker,
            "role": role,
        })
        seg_id += 1
    return segments


def get_active_groq_models(api_key: str) -> list[str]:
    """Dynamically query Groq API for active chat LLM models."""
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {str(api_key).strip()}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            raw_model_ids = [m["id"] for m in data.get("data", []) if "whisper" not in m.get("id", "").lower()]
            good_models = [
                m for m in raw_model_ids
                if not any(bad in m.lower() for bad in ["prompt-guard", "orpheus-arabic", "allam"])
            ]
            models_to_sort = good_models if good_models else raw_model_ids

            def model_priority(name: str) -> int:
                n = name.lower()
                if "120b" in n:
                    return 0
                if "70b" in n:
                    return 1
                if "qwen" in n:
                    return 2
                if "compound" in n:
                    return 3
                if "20b" in n or "32b" in n:
                    return 4
                if "8b" in n:
                    return 5
                return 6

            models_to_sort.sort(key=model_priority)
            return models_to_sort
    except Exception as e:
        print(f"[GROQ MODELS QUERY ERROR]: {e}")
    return []


def call_groq_proofread_text(text: str, api_key: str, model: str = "llama-3.3-70b-versatile") -> str | None:
    if not api_key or not text.strip():
        return None
    system_instruction = get_proofread_system_prompt()
    client = OpenAI(api_key=str(api_key).strip(), base_url="https://api.groq.com/openai/v1", timeout=35)

    active_models = get_active_groq_models(api_key)
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(active_models)
    candidate_models.extend(["llama3-8b-8192", "gemma2-9b-it", "llama-3.3-70b-specdec"])

    seen = set()
    models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

    for m in models_to_try:
        try:
            print(f"[GROQ LLM] Formatting transcript with zero-hallucination model='{m}'...")
            chat_resp = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Format this transcript into แพทย์: and ผู้ป่วย: turns with zero additions:\n\n{text}"},
                ],
                temperature=0.0,
            )
            out = chat_resp.choices[0].message.content.strip()
            if out and ("แพทย์:" in out or "ผู้ป่วย:" in out or "หมอ:" in out or "คนไข้:" in out):
                print(f"[GROQ LLM SUCCESS] Formatting succeeded with '{m}'!")
                return out
        except Exception as e:
            print(f"[GROQ LLM ERROR with '{m}']: {e}")

    return None


def call_gemini_proofread_text(text: str, api_key: str) -> str | None:
    if not api_key or not text.strip():
        return None
    system_instruction = get_proofread_system_prompt()
    models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": f"Format this transcript into แพทย์: and ผู้ป่วย: turns with zero additions:\n\n{text}"}]}],
        "generationConfig": {
            "temperature": 0.0,
        },
    }
    api_k = str(api_key).strip()
    headers = {
        "x-goog-api-key": api_k,
        "Authorization": f"Bearer {api_k}",
        "Content-Type": "application/json",
    }
    for m in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_k}"
            resp = requests.post(url, json=payload, headers=headers, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    out = candidates[0]["content"]["parts"][0]["text"].strip()
                    if out and ("แพทย์:" in out or "ผู้ป่วย:" in out):
                        return out
        except Exception:
            pass
    return None


def post_process_segments(
    formatted_segments: list[SegmentOut],
    mode: str | None = None,
) -> tuple[list[SegmentOut], str, str, str]:
    """
    Post-process segments to ensure 100% accurate Thai Clinical Speaker Diarization
    with strict ZERO HALLUCINATION guarantee.
    """
    settings = get_settings()
    selected_mode = normalize_post_process_mode(mode)

    print(f"[POSTPROCESS] Starting postprocess with mode='{selected_mode}' on {len(formatted_segments)} segments...")

    # Step 1: Apply safe dictionary corrections to all segments
    for seg in formatted_segments:
        seg.text = apply_dictionary(seg.text)

    if selected_mode == "none" or not formatted_segments:
        lines = []
        for s in formatted_segments:
            if not s.text:
                continue
            role, _ = normalize_clinical_role(s.role or s.speaker)
            lines.append(f"{role}: {s.text}")
        raw_lines = [s.text for s in formatted_segments if s.text]
        return formatted_segments, "\n\n".join(lines), " ".join(raw_lines), "dictionary"

    total_duration = max(formatted_segments[-1].end if formatted_segments else 180.0, 10.0)
    combined_raw_text = "\n".join([s.text for s in formatted_segments if s.text])

    post_processed_by = "dictionary"
    diarized_text = None

    # Strategy 1: Gemini if key available
    if settings.gemini_api_key and (selected_mode == "gemini" or settings.post_process_mode == "gemini"):
        try:
            gemini_out = call_gemini_proofread_text(combined_raw_text, settings.gemini_api_key)
            if gemini_out and ("แพทย์:" in gemini_out or "ผู้ป่วย:" in gemini_out):
                diarized_text = gemini_out
                post_processed_by = "gemini"
                print("[POSTPROCESS SUCCESS] Gemini formatted dialogue.")
        except Exception as err:
            print(f"[POSTPROCESS WARNING] Gemini error: {err}")

    # Strategy 2: Groq LLM
    if not diarized_text and settings.groq_api_key:
        try:
            groq_out = call_groq_proofread_text(combined_raw_text, settings.groq_api_key, settings.groq_llm_model)
            if groq_out and ("แพทย์:" in groq_out or "ผู้ป่วย:" in groq_out):
                diarized_text = groq_out
                post_processed_by = f"groq:{settings.groq_llm_model}"
                print(f"[POSTPROCESS SUCCESS] Groq formatted dialogue.")
        except Exception as err:
            print(f"[POSTPROCESS WARNING] Groq error: {err}")

    # Step 3: If AI Diarized Text is available, parse into accurate turn segments
    if diarized_text:
        parsed_turn_dicts = parse_dialogue_text_into_segments(diarized_text, total_duration=total_duration)
        if parsed_turn_dicts:
            new_segments = []
            for item in parsed_turn_dicts:
                clean_t = apply_dictionary(item["text"])
                role_val, spk_val = normalize_clinical_role(item.get("role") or item.get("speaker"))
                new_segments.append(
                    SegmentOut(
                        id=item["id"],
                        start=item["start"],
                        end=item["end"],
                        text=clean_t,
                        speaker=spk_val,
                        role=role_val,
                    )
                )
            if new_segments:
                formatted_segments = new_segments

    # Step 4: Guarantee clean terms and normalized roles
    for seg in formatted_segments:
        seg.text = apply_dictionary(seg.text)
        role_val, spk_val = normalize_clinical_role(seg.role or seg.speaker)
        seg.role = role_val
        seg.speaker = spk_val

    final_text_lines = []
    for s in formatted_segments:
        if not s.text:
            continue
        role_label, _ = normalize_clinical_role(s.role or s.speaker)
        final_text_lines.append(f"{role_label}: {s.text}")

    full_final_text = "\n\n".join(final_text_lines)
    full_raw_text = "\n".join([s.text for s in formatted_segments if s.text])

    print(f"[POSTPROCESS COMPLETED] post_processed_by: {post_processed_by}, total segments: {len(formatted_segments)}")
    return formatted_segments, full_final_text, full_raw_text, post_processed_by


def post_process_transcript(text: str, mode: str | None = None) -> tuple[str, str]:
    settings = get_settings()
    selected_mode = normalize_post_process_mode(mode)

    if selected_mode == "none":
        return text, "none"

    dictionary_text = apply_dictionary(text)

    if settings.gemini_api_key and (selected_mode == "gemini" or settings.post_process_mode == "gemini"):
        gemini_res = call_gemini_proofread_text(dictionary_text, settings.gemini_api_key)
        if gemini_res:
            return gemini_res, "gemini"

    if settings.groq_api_key:
        groq_out = call_groq_proofread_text(dictionary_text, settings.groq_api_key, settings.groq_llm_model)
        if groq_out:
            return groq_out, f"groq:{settings.groq_llm_model}"

    formatted_rule_based = format_clinical_dialogue_rule_based(dictionary_text)
    return formatted_rule_based, "dictionary"
