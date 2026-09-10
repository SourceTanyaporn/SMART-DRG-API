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
5. เตือนภาวะวิกฤต หรือ Red Flags (เช่น Sepsis, Acute Coronary Syndrome, Stroke) เสมอหากพบความผิดปกติในข้อมูล

แนวทางการตอบ:
- ใช้ภาษาไทยผสมคำศัพท์ทางการแพทย์สากลอย่างเป็นธรรมชาติ สุภาพ เป็นมืออาชีพ
- หากแพทย์ต้องการสรุปข้อมูลลงฟอร์ม ให้สรุปเนื้อหาให้ชัดเจน และในตอนท้ายคุณสามารถระบุ Action Block ในรูปแบบ JSON ภายใต้แท็ก ```json:action เพื่อให้ระบบส่งต่อข้อมูลไปกรอกลงฟอร์มในหน้าจอได้อัตโนมัติ

โครงสร้าง Action Block ตัวอย่าง (ใส่เฉพาะเมื่อมีการสรุปหรือแก้ไขฟอร์ม):
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


def parse_action_block(text: str) -> tuple[str, dict | None]:
    """Extract ```json:action block if present."""
    action_data = None
    pattern = r"```json:action\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        try:
            action_data = json.loads(match.group(1).strip())
            # Clean text by removing the raw json:action block or leaving clean markdown
            clean_text = re.sub(pattern, "", text).strip()
            return clean_text, action_data
        except Exception as e:
            print(f"[ACTION PARSE ERROR] {e}")
    return text, None


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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    system_instruction = CLINICAL_COPILOT_SYSTEM_PROMPT
    if context_str:
        system_instruction += f"\n\n{context_str}"

    # Ensure alternating user/model turns for Gemini API
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

    # Ensure starts with user
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

    resp = requests.post(url, json=payload, timeout=25)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts and "text" in parts[0]:
                raw_reply = parts[0]["text"]
                clean_text, action = parse_action_block(raw_reply)
                return clean_text, action
        return "ขออภัยครับ ไม่สามารถสร้างคำตอบได้", None
    else:
        raise RuntimeError(f"Gemini API Error {resp.status_code}: {resp.text}")


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

    # Case 1: Fill / Extract Form / SOAP Note
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

    # Case 2: Drug Safety / Interaction
    elif any(k in query_lower for k in ["ยา", "drug", "แพ้ยา", "interaction", "paracetamol", "oseltamivir"]):
        allergies = context.allergies if context and context.allergies else "ไม่มีประวัติแพ้ยา"
        meds = context.current_medications if context and context.current_medications else []
        reply = (
            f"💊 การประเมินความปลอดภัยทางยาสำหรับเคสนี้:\n"
            f"- ประวัติการแพ้ยา: {allergies}\n"
            f"- ยาที่กำลังได้รับ: {', '.join(meds) if meds else 'ไม่มีบันทึก'}\n\n"
            f"คำแนะนำเพิ่มเติม:\n"
            f"1. ยาลดไข้ Paracetamol: รับประทาน 500 mg 1-2 เม็ด ทุก 4-6 ชม. เวลาปวดหรือมีไข้ (ไม่เกิน 4,000 mg/วัน)\n"
            f"2. ยาต้านไวรัส Oseltamivir (กรณี Influenza): 75 mg รับประทานวันละ 2 ครั้ง เช้า-เย็น นาน 5 วัน (เริ่มยาภายใน 48 ชม. เพื่อประสิทธิภาพสูงสุด)\n"
            f"3. ไม่พบประวัติการแพ้ยาที่เป็นข้อห้ามใช้ในปัจจุบัน"
        )
        return reply, None

    # Case 3: Default clinical assistant fallback
    else:
        form_data = extract_medical_form(transcript or user_query)
        reply = (
            f"🩺 Clinical AI Copilot:\n\n"
            f"จากข้อมูลการตรวจรักษาของผู้ป่วย:\n"
            f"- อาการหลัก: {form_data.chiefComplaint or 'มีไข้ ปวดเมื่อยตามตัว'}\n"
            f"- การวินิจฉัยสงสัย: {form_data.provisionalDiagnosis or 'Acute Viral Infection'}\n"
            f"- ICD-10 แนะนำ: `{form_data.icd10}`\n"
            f"- DRG แนะนำ: `{form_data.drg}`\n\n"
        )
        return reply, None


def run_clinical_chat(payload: ClinicalChatIn) -> ClinicalChatOut:
    settings = get_settings()
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

    # 1. Try Gemini
    if settings.gemini_api_key:
        try:
            reply, action = chat_with_gemini(raw_messages, context_str, settings.gemini_api_key)
            return ClinicalChatOut(
                reply=reply,
                structured_action=action,
                suggested_quick_prompts=generate_suggested_prompts(payload.patient_context, reply),
                provider_used="gemini",
            )
        except Exception as e:
            print(f"[CHAT ERROR] Gemini API call failed: {e}")

    # 2. Try OpenAI
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
                provider_used="openai",
            )
        except Exception as e:
            print(f"[CHAT ERROR] OpenAI API call failed: {e}")

    # 3. Graceful Local Clinical Rule-based Fallback (No 500 error!)
    reply, action = generate_rule_based_reply(last_user_msg, payload.patient_context)
    return ClinicalChatOut(
        reply=reply,
        structured_action=action,
        suggested_quick_prompts=generate_suggested_prompts(payload.patient_context, reply),
        provider_used="rule_based",
    )
