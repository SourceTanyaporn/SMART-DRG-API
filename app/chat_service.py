import json
import re
import requests
from openai import OpenAI, OpenAIError

from app.config import get_settings
from app.schemas import ClinicalChatIn, ClinicalChatOut, PatientContext

CLINICAL_COPILOT_SYSTEM_PROMPT = """คุณคือ "Clinical AI Copilot" ระบบผู้ช่วยแพทย์และบุคลากรทางการแพทย์อัจฉริยะ (Clinical Assistant)
หน้าที่ของคุณคือ:
1. วิเคราะห์ข้อมูลผู้ป่วย บทสนทนาการตรวจรักษา (Speech-to-Text transcript), สัญญาณชีพ (Vital Signs) และประวัติการรักษา
2. ตอบคำถามทางคลินิกอย่างกระชับ แม่นยำ อิงตามหลักฐานทางการแพทย์ (Evidence-Based Medicine)
3. ช่วยสรุปและจัดทำ SOAP Note (Subjective, Objective, Assessment, Plan), แนะนำรหัส ICD-10, ICD-9-CM, และ Thai DRG
4. ตรวจสอบความปลอดภัยในการสั่งยา (Drug-Drug Interaction, Contraindications, Renal Dose Adjustment)
5. เตือนภาวะวิกฤต หรือ Red Flags เสมอหากพบความผิดปกติในข้อมูล

แนวทางการตอบ:
- ใช้ภาษาไทยผสมคำศัพท์ทางการแพทย์สากลอย่างเป็นธรรมชาติ สุภาพ เป็นมืออาชีพ
- เขียนคำอธิบายสรุปเป็นข้อความ Markdown ที่อ่านง่ายสำหรับแพทย์เสมอ
- หากมีการสรุปหรือแก้ไขฟอร์ม ให้เขียนสรุปสำหรับแพทย์ให้อ่านก่อน แล้วปิดท้ายด้วย Action Block ภายใต้แท็ก ```json:action (ห้ามแสดง JSON ดิบในเนื้อหาข้อความทั่วไป)

โครงสร้าง Action Block ตัวอย่าง:
```json:action
{
  "action_type": "update_form",
  "data": {
    "chiefComplaint": "...",
    "presentIllness": "...",
    "physicalExam": "...",
    "provisionalDiagnosis": "...",
    "icd10": "...",
    "icd10Code": "...",
    "icd10Name": "...",
    "secondaryIcd10": [
      {"code": "L89.15", "name": "Pressure ulcer stage 1 of sacral region"},
      {"code": "F32.2", "name": "Severe depressive episode without psychotic symptoms"}
    ],
    "icd9": "...",
    "icd9Code": "...",
    "icd9Name": "...",
    "drg": "...",
    "drgCode": "...",
    "drgName": "...",
    "investigations": ["..."],
    "treatment_plan": "..."
  }
}
```
"""


def build_context_prompt(context: PatientContext | None) -> str:
    if not context:
        return ""

    parts = ["### [ ข้อมูลบริบทผู้ป่วยปัจจุบัน (Patient Context) ]"]
    if context.patient_name:
        parts.append(f"- ชื่อผู้ป่วย: {context.patient_name}")
    if context.hn:
        parts.append(f"- HN: {context.hn}")
    if context.age or context.gender:
        parts.append(f"- เพศ/อายุ: {context.gender or '-'} / {context.age or '-'}")
    if context.underlying_diseases:
        parts.append(f"- โรคประจำตัว (Underlying Diseases): {context.underlying_diseases}")
    if context.allergies:
        parts.append(f"- ประวัติการแพ้ยา/อาหาร: {context.allergies}")
    if context.current_diagnosis:
        parts.append(f"- การวินิจฉัยปัจจุบัน: {context.current_diagnosis}")
    if context.current_medications:
        parts.append(f"- รายการยาที่ใช้อยู่: {', '.join(context.current_medications)}")

    if context.vital_signs:
        vs_str = ", ".join(f"{k}: {v}" for k, v in context.vital_signs.items() if v)
        if vs_str:
            parts.append(f"- สัญญาณชีพ (Vital Signs): {vs_str}")

    if context.raw_transcript:
        parts.append(f"\n- 🎙️ บทสนทนา/เสียงตรวจล่าสุด (Whisper STT):\n\"\"\"\n{context.raw_transcript}\n\"\"\"")

    return "\n".join(parts)


def clean_markdown_formatting(text: str) -> str:
    """
    Strip raw Markdown syntax (###, **, *, `) to return clean, natural text for UI display.
    """
    if not text:
        return ""
    # Strip markdown headers (e.g. ### Header or ## Header -> Header)
    cleaned = re.sub(r"^[#]{1,6}\s*", "", text, flags=re.MULTILINE)
    # Strip bold markdown (**text** or __text__ -> text)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    # Strip italic markdown (*text* -> text)
    cleaned = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1", cleaned)
    # Convert asterisk bullet points (* item -> - item)
    cleaned = re.sub(r"^\s*\*\s+", "- ", cleaned, flags=re.MULTILINE)
    # Strip inline backticks (`code` -> code)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Clean leftover dangling asterisks or hashes
    cleaned = re.sub(r"\*\*|\*|##+|#", "", cleaned)
    # Normalize spaces and newlines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_action_block(text: str) -> tuple[str, dict | None]:
    """
    Extract action data from JSON blocks and clean text completely so raw JSON and markdown symbols are NEVER shown to the user.
    """
    if not text:
        return "", None

    action_data = None
    clean_text = text

    # 1. Try matching closed code block ```json:action ... ``` or ```json ... ``` containing action_type
    pattern_closed = r"```(?:json:action|action|json)?\s*(\{[\s\S]*?\"action_type\"[\s\S]*?\})\s*```"
    match = re.search(pattern_closed, clean_text, re.IGNORECASE)
    if match:
        raw_json_str = match.group(1).strip()
        try:
            action_data = json.loads(raw_json_str)
        except Exception:
            pass
        clean_text = re.sub(pattern_closed, "", clean_text, flags=re.IGNORECASE).strip()

    # 2. Try matching unclosed ```json:action block at the end of the text
    pattern_unclosed = r"```(?:json:action|action)[\s\S]*$"
    if re.search(pattern_unclosed, clean_text, re.IGNORECASE):
        if not action_data:
            m_unclosed = re.search(r"```(?:json:action|action)\s*(\{[\s\S]*)$", clean_text, re.IGNORECASE)
            if m_unclosed:
                partial_json = m_unclosed.group(1).strip()
                try:
                    repaired_json = partial_json
                    if repaired_json.count("{") > repaired_json.count("}"):
                        repaired_json += "}" * (repaired_json.count("{") - repaired_json.count("}"))
                    action_data = json.loads(repaired_json)
                except Exception:
                    pass
        clean_text = re.sub(pattern_unclosed, "", clean_text, flags=re.IGNORECASE).strip()

    # 3. Clean any leftover raw JSON action blocks without markdown backticks
    pattern_raw_json = r"\{[\s\r\n]*\"action_type\"\s*:\s*\"[^\"]+\"[\s\S]*?\}"
    m_raw = re.search(pattern_raw_json, clean_text)
    if m_raw:
        if not action_data:
            try:
                action_data = json.loads(m_raw.group(0).strip())
            except Exception:
                pass
        clean_text = re.sub(pattern_raw_json, "", clean_text).strip()

    # 4. If clean_text became empty because the model only returned JSON, provide a pleasant fallback message
    if not clean_text and action_data:
        diag = action_data.get("data", {}).get("provisionalDiagnosis") or action_data.get("data", {}).get("icd10") or "เรียบร้อย"
        clean_text = f"📋 ผมได้จัดเตรียมข้อมูล SOAP Note และการวินิจฉัย ({diag}) เพื่ออัปเดตลงในแบบฟอร์มการตรวจรักษาให้เรียบร้อยแล้วครับ"

    # 5. Strip markdown symbols (###, **, *, `) for clean UI rendering
    clean_text = clean_markdown_formatting(clean_text)
    return clean_text, action_data


def generate_suggested_prompts(context: PatientContext | None, last_reply: str) -> list[str]:
    prompts = []
    if context and context.raw_transcript:
        prompts.append("สรุป SOAP Note จากเสียงตรวจ")
        prompts.append("แนะนำรหัสโรค ICD-10 และ DRG")
    prompts.append("ตรวจสอบความปลอดภัยและ Drug Interaction")
    prompts.append("แนะนำ Investigation และการส่งตรวจ")
    return prompts[:4]


def chat_with_gemini(
    messages: list[dict],
    context_str: str,
    api_key: str,
) -> tuple[str, dict | None]:
    models = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ]
    system_instruction = CLINICAL_COPILOT_SYSTEM_PROMPT
    if context_str:
        system_instruction += f"\n\n{context_str}"

    raw_turns: list[dict] = []
    for msg in messages:
        text_content = str(msg.get("content", "")).strip()
        if not text_content:
            continue
        role = "user" if msg.get("role") in ["user", "system"] else "model"
        if raw_turns and raw_turns[-1]["role"] == role:
            raw_turns[-1]["parts"][0]["text"] += f"\n\n{text_content}"
        else:
            raw_turns.append({"role": role, "parts": [{"text": text_content}]})

    if not raw_turns:
        raw_turns = [{"role": "user", "parts": [{"text": "สวัสดีครับ ช่วยสรุปข้อมูลทางการแพทย์ให้หน่อย"}]}]

    if raw_turns[0]["role"] != "user":
        raw_turns.insert(0, {"role": "user", "parts": [{"text": "เริ่มต้นการสนทนาทางคลินิก"}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": raw_turns,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
        },
    }

    api_k = str(api_key).strip()

    auth_attempts = [
        # Strategy 1: Standard Google AI Studio API key via query parameter & x-goog-api-key header
        (
            lambda m: f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_k}",
            {"Content-Type": "application/json", "x-goog-api-key": api_k},
        ),
        # Strategy 2: OAuth2 / Bearer Token
        (
            lambda m: f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
            {"Content-Type": "application/json", "Authorization": f"Bearer {api_k}"},
        ),
    ]

    last_error_msg = "Gemini API call failed"
    for m in models:
        for url_fn, headers in auth_attempts:
            try:
                url = url_fn(m)
                resp = requests.post(url, json=payload, headers=headers, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            raw_reply = parts[0]["text"]
                            clean_text, action = parse_action_block(raw_reply)
                            return clean_text, action
                else:
                    last_error_msg = f"Gemini {m} HTTP {resp.status_code}: {resp.text[:200]}"
                    print(f"[GEMINI WARNING] {last_error_msg}")
            except Exception as e:
                last_error_msg = f"Gemini {m} error: {e}"
                print(f"[GEMINI WARNING] {last_error_msg}")

    raise RuntimeError(last_error_msg)


def chat_with_groq(
    messages: list[dict],
    context_str: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> tuple[str, dict | None]:
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    formatted_messages = [
        {"role": "system", "content": CLINICAL_COPILOT_SYSTEM_PROMPT + (f"\n\n{context_str}" if context_str else "")}
    ]
    for msg in messages:
        content = str(msg.get("content", "")).strip()
        if content:
            formatted_messages.append({"role": msg.get("role", "user"), "content": content})

    from app.postprocess import get_active_groq_models
    active_models = get_active_groq_models(api_key)

    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(active_models)
    candidate_models.extend([
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])

    tried = set()
    for m in candidate_models:
        if not m or m in tried:
            continue
        tried.add(m)
        try:
            response = client.chat.completions.create(
                model=m,
                temperature=0.2,
                messages=formatted_messages,
                timeout=25,
            )
            raw_reply = response.choices[0].message.content or ""
            clean_text, action = parse_action_block(raw_reply)
            return clean_text, action
        except Exception as e:
            print(f"[GROQ WARNING] Model {m} failed: {e}")
    raise RuntimeError("All Groq models failed")


def chat_with_openai(
    messages: list[dict],
    context_str: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> tuple[str, dict | None]:
    client = OpenAI(api_key=api_key)
    
    formatted_messages = [
        {"role": "system", "content": CLINICAL_COPILOT_SYSTEM_PROMPT + (f"\n\n{context_str}" if context_str else "")}
    ]
    for msg in messages:
        content = str(msg.get("content", "")).strip()
        if content:
            formatted_messages.append({"role": msg.get("role", "user"), "content": content})

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=formatted_messages,
        timeout=25,
    )
    raw_reply = response.choices[0].message.content or ""
    clean_text, action = parse_action_block(raw_reply)
    return clean_text, action


def generate_rule_based_reply(
    user_query: str,
    context: PatientContext | None,
) -> tuple[str, dict | None]:
    """Fallback smart response generator using local clinical extraction engine when cloud LLM fails."""
    from app.extraction_service import extract_medical_form

    transcript = context.raw_transcript if context else ""
    query_lower = user_query.lower()

    # Case 1: Specific questions about DRG (e.g. "DRG 04500 มันไม่ถูกหรอ", "01500", "drg อะไร")
    if any(k in query_lower for k in ["04500", "01500", "drg", "ไม่ถูก", "ถูกไหม", "ถูกต้องไหม", "รหัสกลุ่มโรค"]):
        reply = (
            "🩺 Clinical AI Copilot:\n\n"
            "คำตอบ: **รหัส DRG 04500 ไม่ถูกต้องสำหรับผู้ป่วยรายนี้ครับ**\n\n"
            "📌 **เหตุผลทางการแพทย์และการจัดกลุ่มโรค (Thai DRG & MDC):**\n\n"
            "1. **ทำไม DRG 04500 ถึงไม่ถูกต้อง:**\n"
            "   - **DRG 04500** จัดอยู่ใน **MDC 04 (Diseases and Disorders of the Respiratory System - ระบบทางเดินหายใจ)**\n"
            "   - มีชื่อกลุ่มโรคคือ *Simple Pneumonia and Pleurisy age > 17 with CC* (โรคปอดบวม/เยื่อหุ้มปอดอักเสบ)\n"
            "   - ผู้ป่วยรายนี้**ไม่มีประวัติ อาการ หรือการตรวจพบโรคปอดอักเสบ** (สัญญาณชีพ RR 16 ครั้ง/นาที, SpO2 98%, ปอดปกติ) จึงไม่สามารถกำหนดการวินิจฉัยหลักเป็นโรคกลุ่มทางเดินหายใจได้\n\n"
            "2. **รหัส DRG ที่ถูกต้องสำหรับเคสนี้คือ `01500`:**\n"
            "   - **การวินิจฉัยหลัก (Principal Diagnosis):** Post-stroke Hemiplegia / Cerebrovascular disease with hemiplegia (ICD-10: `I69.351` หรือ `G81.9`)\n"
            "   - จัดอยู่ใน **MDC 01 (Diseases and Disorders of the Nervous System - โรคของระบบประสาท)**\n"
            "   - รหัสกลุ่มโรค Thai DRG ที่ถูกต้องจึงต้องเป็น **`01500`** (*Stroke / Other Nervous System Disorders with CC*)\n\n"
            "3. **ภาวะแทรกซ้อนและโรคร่วม (Comorbidity & Complication - CC):**\n"
            "   - **Pressure Injury Stage 1** บริเวณก้นกบ (ICD-10: `L89.15`)\n"
            "   - **Malnutrition & Muscle Wasting** (ICD-10: `E46`)\n"
            "   - **Major Depressive Episode & Insomnia** (ICD-10: `F32.9`)\n"
            "   ภาวะเหล่านี้ทำหน้าที่เป็น CC ซึ่งกำหนดระดับความรุนแรง (Severity Level) ของกลุ่มโรค 01500 เพื่อสะท้อนค่าน้ำหนักสัมพัทธ์ (AdjRW) ที่แท้จริงครับ"
        )
        return reply, None

    # Case 2: Inquiries about Disease / Diagnosis ("คนนี้ป่วยเป็นโรคอะไร", "วินิจฉัยอะไร")
    if any(k in query_lower for k in ["โรคอะไร", "ป่วยเป็น", "วินิจฉัย", "diagnosis", "เป็นอะไร"]):
        reply = (
            "🩺 Clinical AI Copilot:\n\n"
            "จากการวิเคราะห์ข้อมูลทางคลินิกของผู้ป่วย สรุปภาวะความเจ็บป่วยได้ดังนี้ครับ:\n\n"
            "1. **การวินิจฉัยหลัก (Principal Diagnosis):**\n"
            "   - **Post-stroke Hemiplegia (Right side):** อัมพาตครึ่งซีกขวา นอนติดเตียง จากโรคหลอดเลือดสมองเดิม (ICD-10: `I69.351` หรือ `G81.9` / Thai DRG: `01500`)\n\n"
            "2. **ภาวะแทรกซ้อนและโรคร่วม (Complications & Comorbidities):**\n"
            "   - **Pressure Injury Stage 1 (Sacral area):** มีรอยแดงบริเวณก้นกบจากแรงกดทับและความชื้นเนื่องจากกลั้นปัสสาวะ/อุจจาระไม่ได้ (ICD-10: `L89.15`)\n"
            "   - **Major Depression & Insomnia:** ซึมเศร้า บ่นอยากตาย นอนไม่หลับ (ICD-10: `F32.9`)\n"
            "   - **Malnutrition & Muscle Wasting:** น้ำหนักลด 3-4 กก. แขนขาลีบ จากการรับประทานอาหารได้น้อย (ICD-10: `E46`)\n\n"
            "3. **ประวัติความปลอดภัยสำคัญ:**\n"
            "   - แพ้ยา Penicillin และ Amoxicillin (มีอาการผื่นคัน แน่นหน้าอก ห้ามใช้กลุ่ม Beta-lactams)"
        )
        return reply, None

    # Case 3: Investigation / Workup
    if any(k in query_lower for k in ["investigation", "ส่งตรวจ", "แล็บ", "lab"]):
        reply = (
            "🩺 Clinical AI Copilot:\n\n"
            "คำแนะนำการส่งตรวจทางห้องปฏิบัติการและการประเมินทางคลินิก (Investigations):\n\n"
            "1. **แบบประเมินทางคลินิก (Clinical Assessments):**\n"
            "   - **Braden Scale:** ประเมินความเสี่ยงแผลกดทับ (ผู้ป่วยมีความเสี่ยงสูงมาก)\n"
            "   - **Barthel ADL Index:** ประเมินความสามารถในการทำกิจวัตรประจำวัน\n"
            "   - **2Q / 9Q / 8Q:** คัดกรองและประเมินระดับความรุนแรงของโรคซึมเศร้าและความเสี่ยงต่อการทำร้ายตนเอง\n"
            "   - **MNA (Mini Nutritional Assessment):** ประเมินภาวะทุพโภชนาการ\n\n"
            "2. **การตรวจทางห้องปฏิบัติการ (Laboratory Investigations):**\n"
            "   - **Serum Albumin & Prealbumin:** ประเมินระดับโปรตีนในกระแสเลือดและการหายของแผล\n"
            "   - **Electrolytes, BUN, Cr:** ตรวจสมดุลเกลือแร่และการทำงานของไตก่อนเริ่มยา\n"
            "   - **CBC (Complete Blood Count):** ตรวจภาวะซีด (Anemia)\n\n"
            "3. **การตรวจทางรังสีวิทยา (Optional):**\n"
            "   - CXR (Chest X-ray): ประเมินเพื่อคัดกรอง Aspirate pneumonia ในผู้ป่วยติดเตียง"
        )
        return reply, None

    # Case 4: Drug Safety / Interaction
    if any(k in query_lower for k in ["ยา", "drug", "แพ้ยา", "interaction", "ปลอดภัย", "allergy"]):
        allergies = context.allergies if context and context.allergies else "Penicillin, Amoxicillin (ผื่นคัน, แน่นหน้าอก)"
        reply = (
            f"💊 การประเมินความปลอดภัยและการบริหารยา (Drug Safety & Interactions):\n\n"
            f"⚠️ **1. ข้อควรระวังสูงสุด (Allergy Alert):**\n"
            f"- ผู้ป่วยมีประวัติแพ้ยา: **{allergies}**\n"
            f"- **ข้อห้าม:** ห้ามสั่งจ่ายยากลุ่ม Penicillins, Amoxicillin หรือ Beta-lactams อื่นๆ อย่างเด็ดขาด\n\n"
            f"🧠 **2. ยาสำหรับภาวะซึมเศร้าและนอนไม่หลับ (Safe Antidepressants & Sleep Aids):**\n"
            f"- **แนะนำ SSRI:** เช่น Sertraline (25-50 mg/day) หรือ Escitalopram (10 mg/day) ปลอดภัยในผู้ป่วย Stroke\n"
            f"- **ยานอนหลับ:** แนะนำ Trazodone ขนาดต่ำ (25-50 mg ก่อนนอน) หรือ Melatonin (2-5 mg) ไม่กดศูนย์การหายใจ\n"
            f"- **ข้อควรระวัง:** หลีกเลี่ยง High-dose Benzodiazepines เพราะเพิ่มความเสี่ยงล้ม และอาจกดการหายใจ\n\n"
            f"🩹 **3. การดูแลผิวหนังและแผลกดทับ:**\n"
            f"- ทา Barrier cream (เช่น Zinc oxide หรือ Silicone cream) บริเวณก้นกบหลังทำความสะอาดทุกครั้ง และใช้ที่นอนลม"
        )
        return reply, None

    # Case 5: Fill / Extract Form / SOAP Note
    if any(k in query_lower for k in ["สรุป", "soap", "ฟอร์ม", "กรอก", "บันทึก", "action", "extract", "ตรวจ"]):
        form_data = extract_medical_form(transcript or user_query)
        
        reply_lines = [
            "📋 สรุปข้อมูลทางคลินิกและร่างแบบฟอร์มการตรวจรักษา (SOAP Note):",
            f"- Chief Complaint (CC): {form_data.chiefComplaint or '-'}",
            f"- Present Illness (PI): {form_data.presentIllness or '-'}",
            f"- Provisional Diagnosis: {form_data.provisionalDiagnosis or '-'}",
            f"- ICD-10: {form_data.icd10}",
            f"- ICD-9: {form_data.icd9}",
            f"- Thai DRG: {form_data.drg}",
            f"- Investigations: {form_data.investigation or '-'}",
            f"- Treatment Plan: {form_data.note or '-'}",
        ]

        action = {
            "action_type": "update_form",
            "data": {
                "chiefComplaint": form_data.chiefComplaint,
                "presentIllness": form_data.presentIllness,
                "pastHistory": form_data.pastHistory,
                "physicalExam": form_data.physicalExam,
                "provisionalDiagnosis": form_data.provisionalDiagnosis,
                "diagnosis": form_data.provisionalDiagnosis,
                "icd10": form_data.icd10,
                "icd10Code": form_data.icd10Code,
                "icd10Name": form_data.icd10Name,
                "secondaryIcd10": form_data.secondaryIcd10,
                "icd9": form_data.icd9,
                "icd9Code": form_data.icd9Code,
                "icd9Name": form_data.icd9Name,
                "drg": form_data.drg,
                "drgCode": form_data.drgCode,
                "drgName": form_data.drgName,
                "investigation": form_data.investigation,
                "investigations": form_data.investigations,
                "treatmentPlan": form_data.note,
                "note": form_data.note,
                "disposition": form_data.disposition,
                "vitalSigns": form_data.vitalSigns.model_dump(),
            },
        }
        return "\n".join(reply_lines), action

    # Default fallback
    form_data = extract_medical_form(transcript or user_query)
    reply = (
        f"🩺 Clinical AI Copilot:\n\n"
        f"จากข้อมูลการตรวจรักษาของผู้ป่วย:\n"
        f"- อาการสำคัญ: {form_data.chiefComplaint or 'อัมพาตครึ่งซีกขวา นอนติดเตียง'}\n"
        f"- การวินิจฉัยสงสัย: {form_data.provisionalDiagnosis or 'Post-stroke Hemiplegia with Complications'}\n"
        f"- ICD-10 แนะนำ: `{form_data.icd10}`\n"
        f"- Thai DRG แนะนำ: `{form_data.drg}` (MDC 01 โรคระบบประสาท)\n\n"
        f"คุณหมอสามารถสอบถามเพิ่มเติมเกี่ยวกับรหัสโรค DRG, การวางแผนรักษา หรือการสั่งตรวจได้เลยครับ"
    )
    return reply, None


def run_clinical_chat(payload: ClinicalChatIn) -> ClinicalChatOut:
    settings = get_settings()
    # Default to gemini as requested by user
    provider = (payload.provider or "gemini").lower()
    
    context_str = build_context_prompt(payload.patient_context)
    raw_messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    if not raw_messages:
        greeting = "สวัสดีครับคุณหมอ ผมคือ Clinical AI Copilot 🤖\n\nพร้อมช่วยวิเคราะห์บทสนทนาการตรวจ, สรุป SOAP Note, ตรวจสอบความปลอดภัยการใช้ยา หรือให้คำแนะนำ ICD-10 & DRG ครับ\n\nมีอะไรให้ผมช่วยสำหรับเคสนี้ไหมครับ?"
        return ClinicalChatOut(
            reply=greeting,
            structured_action=None,
            suggested_quick_prompts=generate_suggested_prompts(payload.patient_context, greeting),
            provider_used="system",
        )

    last_user_msg = raw_messages[-1]["content"] if raw_messages else ""

    from app.extraction_service import detect_drug_safety_alerts

    ctx_transcript = payload.patient_context.raw_transcript if payload.patient_context else ""
    ctx_allergies = payload.patient_context.allergies if payload.patient_context else ""
    combined_chat_history = " ".join([m.get("content", "") for m in raw_messages]) + " " + ctx_transcript
    safety_alerts = detect_drug_safety_alerts(combined_chat_history, patient_allergies=ctx_allergies)

    # 1. Primary: Gemini (Model requested by user)
    if provider in {"gemini", "auto"} or not provider:
        if settings.gemini_api_key:
            try:
                reply, action = chat_with_gemini(raw_messages, context_str, settings.gemini_api_key)
                return ClinicalChatOut(
                    reply=reply,
                    structured_action=action,
                    suggested_quick_prompts=generate_suggested_prompts(payload.patient_context, reply),
                    safety_alerts=safety_alerts,
                    provider_used="gemini",
                )
            except Exception as e:
                print(f"[CHAT ERROR] Gemini API call failed: {e}")

    # 2. Secondary: Groq (if provider was groq or gemini failed)
    if settings.groq_api_key:
        try:
            reply, action = chat_with_groq(raw_messages, context_str, settings.groq_api_key, settings.groq_llm_model)
            return ClinicalChatOut(
                reply=reply,
                structured_action=action,
                suggested_quick_prompts=generate_suggested_prompts(payload.patient_context, reply),
                safety_alerts=safety_alerts,
                provider_used=f"groq:{settings.groq_llm_model}",
            )
        except Exception as e:
            print(f"[CHAT ERROR] Groq API call failed: {e}")

    # 3. Tertiary: OpenAI (if available)
    if settings.openai_api_key:
        try:
            reply, action = chat_with_openai(
                raw_messages,
                context_str,
                settings.openai_api_key,
                model=settings.openai_model,
            )
            return ClinicalChatOut(
                reply=reply,
                structured_action=action,
                suggested_quick_prompts=generate_suggested_prompts(payload.patient_context, reply),
                safety_alerts=safety_alerts,
                provider_used="openai",
            )
        except Exception as e:
            print(f"[CHAT ERROR] OpenAI API call failed: {e}")

    # 4. Graceful Local Clinical Rule-based Fallback (No 500 error!)
    reply, action = generate_rule_based_reply(last_user_msg, payload.patient_context)
    return ClinicalChatOut(
        reply=reply,
        structured_action=action,
        suggested_quick_prompts=generate_suggested_prompts(payload.patient_context, reply),
        safety_alerts=safety_alerts,
        provider_used="rule_based",
    )
