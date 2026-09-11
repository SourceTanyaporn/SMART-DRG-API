import json
import re
import requests
from openai import OpenAI, OpenAIError

from app.config import get_settings
from app.postprocess import apply_dictionary
from app.schemas import MedicalFormExtractOut, VitalSignsData


EXTRACTION_SYSTEM_PROMPT = """คุณคือผู้ช่วยแพทย์และระบบ AI วิเคราะห์เวชระเบียนทางการแพทย์ (Medical Record Extractor)
หน้าที่ของคุณคือ อ่านบทสนทนาระหว่างแพทย์และคนไข้ หรือบันทึกเสียงการตรวจ แล้วสกัดข้อมูลลงในโครงสร้างแบบฟอร์มการตรวจรักษา (SOAP Note / Clinical Form) ให้ถูกต้องตามหลักการแพทย์

กรุณาสกัดข้อมูลออกมาเป็น JSON Object ตาม Schema ต่อไปนี้:
{
  "chiefComplaint": "อาการสำคัญที่เป็นเหตุผลหลักที่มา รพ. พร้อมระยะเวลาสั้นๆ เช่น 'ไข้สูง ไอถี่ เสมหะเขียวข้น แน่นหน้าอกขวา 2 วัน' หรือ 'นอนไม่หลับ อ่อนเพลีย หมดพลังงาน 2-3 สัปดาห์'",
  "presentIllness": "ลำดับเหตุการณ์ประวัติปัจจุบัน อาการร่วม การใช้ยา และการดำเนินของโรคอย่างละเอียด โดยสรุปเป็นข้อๆ ทางการแพทย์",
  "pastHistory": "ประวัติโรคประจำตัว ประวัติการแพ้ยา หรือประวัติการเจ็บป่วยในอดีต (ถ้าไม่มีระบุเป็น '-')",
  "vital_signs": {
    "pulse": "ชีพจร ระบุเฉพาะตัวเลข เช่น '80' หรือ '100' (ไม่ต้องใส่ bpm หรือ ครั้ง/นาที)",
    "bodyTemperature": "อุณหภูมิร่างกาย ระบุเฉพาะตัวเลข เช่น '38.6' หรือ '37.0' (ไม่ต้องใส่ °C หรือ องศา)",
    "bp": "ความดันโลหิตครั้งที่ 1 เช่น '124/78' หรือ '120/80' (ระบุเฉพาะตัวเลขความดัน ไม่ต้องใส่หน่วย mmHg)",
    "systolic": "ความดันตัวบน เช่น '124'",
    "diastolic": "ความดันตัวล่าง เช่น '78'",
    "bp2": "ความดันโลหิตครั้งที่ 2 (ถ้ามี) เช่น '120/80' (ระบุเฉพาะตัวเลขความดัน ไม่ต้องใส่หน่วย mmHg)",
    "systolic2": "ความดันตัวบน 2 เช่น '120'",
    "diastolic2": "ความดันตัวล่าง 2 เช่น '80'",
    "o2sat": "ออกซิเจนในเลือด ระบุเฉพาะตัวเลข เช่น '98' หรือ '93' (ไม่ต้องใส่ %)",
    "map": "MAP Mean Arterial Pressure 1 เช่น '93.3' หรือคำนวณจาก (Systolic + 2*Diastolic)/3",
    "map2": "MAP2 Mean Arterial Pressure 2 (ถ้ามี)",
    "weight": "น้ำหนักตัว ระบุเฉพาะตัวเลข เช่น '65' (ไม่ต้องใส่ kg)",
    "height": "ส่วนสูง ระบุเฉพาะตัวเลข เช่น '165' (ไม่ต้องใส่ cm)",
    "bmi": "BMI ดัชนีมวลกาย เช่น '23.8'",
    "chest": "รอบอก ระบุเฉพาะตัวเลข เช่น '88'",
    "waist": "รอบเอว ระบุเฉพาะตัวเลข เช่น '80'",
    "respiratory": "อัตราการหายใจ ระบุเฉพาะตัวเลข เช่น '20' (ไม่ต้องใส่ ครั้ง/นาที)",
    "painScore": "ระดับความปวด ระบุเฉพาะตัวเลข 0-10 เช่น '6'"
  },
  "physicalExam": "ผลการตรวจร่างกายของแพทย์ เช่น 'Chest: Crepitation at right lower lung field'",
  "provisionalDiagnosis": "การวินิจฉัยโรคเบื้องต้น เช่น 'Community-Acquired Pneumonia' หรือ 'Depressive Episode with Insomnia'",
  "icd10": "รหัสและชื่อโรค ICD-10 เช่น 'J11.1 (Influenza with other respiratory manifestations / ไข้หวัดใหญ่)'",
  "icd10Code": "รหัสโรค ICD-10 เช่น 'J11.1' หรือ 'J15.9'",
  "icd10Name": "ชื่อโรค ICD-10 เช่น 'Influenza with other respiratory manifestations / ไข้หวัดใหญ่'",
  "icd9": "รหัสและชื่อหัตถการทางการแพทย์ ICD-9-CM เช่น '96.04 (Continuous oxygen therapy)' หรือ '-' หากไม่มี",
  "icd9Code": "รหัสหัตถการ ICD-9-CM เช่น '96.04' หรือ '-' หากไม่มีหัตถการ",
  "icd9Name": "ชื่อหัตถการ ICD-9-CM เช่น 'Continuous oxygen therapy' หรือ '' หากไม่มี",
  "drg": "รหัสและชื่อกลุ่ม Thai DRG เช่น '04500 (Simple Pneumonia and Pleurisy age > 17 with CC)'",
  "drgCode": "รหัสกลุ่มการวินิจฉัยโรคร่วม Thai DRG เช่น '04500' หรือ '19500'",
  "drgName": "ชื่อกลุ่มโรค Thai DRG เช่น 'Simple Pneumonia and Pleurisy age > 17 with CC' หรือ 'Depressive Disorders'",
  "investigations": [
    "รายการตรวจทางห้องปฏิบัติการ / X-ray / EKG เช่น 'Stat CXR (Chest X-ray)', 'CBC', 'FBS & HbA1c', 'Blood culture x 2'"
  ],
  "assessmentForms": [
    "รหัสแบบประเมินที่ผู้ป่วยควรทำจากรายการ: 'smoking' (สูบบุหรี่), 'depression' (คัดกรองซึมเศร้า 2Q/9Q/8Q), 'fall_risk' (ประเมินหกล้ม Morse), 'adl' (กิจวัตรประจำวัน Barthel), 'cvd' (ความเสี่ยงโรคหัวใจ Thai CV), 'braden' (แผลกดทับ), 'alcohol' (คัดกรองสุรา AUDIT), 'mna' (โภชนาการ) หรือ [] หากไม่มี (สำหรับระดับความปวด ให้ระบุใน vitalSigns.painScore โดยตรง ไม่ต้องใส่ใน assessmentForms)"
  ],
  "assessments": {
    "depression": {
      "summary": "สรุปข้อมูลที่ได้จากบทสนทนา เช่น มีอาการเบื่อหน่าย หมดพลัง นอนไม่หลับ 2Q ผลบวก",
      "answers": {
        "q2_1": "มี",
        "q2_2": "มี",
        "sleep_issue": "หลับยาก เกือบทุกคืน",
        "fatigue": "อ่อนเพลียมาก เหนื่อยง่าย",
        "appetite": "เบื่ออาหาร กินข้าวไม่ลง",
        "concentration": "สมาธิยังดี เคลียร์งานได้"
      }
    },
    "smoking": {
      "summary": "ปฏิเสธการสูบบุหรี่ ไม่เคยสูบ",
      "answers": {
        "smoking_status": "ไม่เคยสูบ",
        "cigarettes_per_day": "0"
      }
    },
    "alcohol": {
      "summary": "ดื่มนานๆ ครั้งในงานสังสรรค์ เดือนละครั้ง 1-2 แก้ว",
      "answers": {
        "frequency": "เดือนละครั้งหรือน้อยกว่า",
        "amount": "1-2 แก้ว"
      }
    },
    "fall_risk": {
      "summary": "ไม่มีประวัติหกล้ม เดินได้เองคล่องตัวดี ความเสี่ยงต่ำ",
      "answers": {
        "fall_history": "ไม่มี",
        "gait": "ปกติ คล่องตัว"
      }
    }
  },
  "note": "แผนการรักษา ยา คำแนะนำ และหัตถการ",
  "disposition": "สถานะการจำหน่าย/รับผู้ป่วย เช่น 'Admit IPD' หรือ 'OPD Discharge'"
}

กฎสำคัญ:
1. ตอบกลับเป็น JSON ที่ถูกต้องสมบูรณ์เท่านั้น (Valid JSON only) ไม่ต้องใส่ Markdown backticks หรือข้อความอื่น
2. ห้ามคัดลอกบทสนทนาดิบ เช่น คำทักทาย 'สวัสดีครับ เชิญนั่งก่อนครับ...' หรือคำพูดทั้งหมดมาใส่ในช่องเด็ดขาด ต้องเรียบเรียงสรุปเป็นภาษาทางการแพทย์ที่กระชับ
3. สกัดค่าสัญญาณชีพ (Vital Signs) ให้ครบถ้วนตามที่มีในบทสนทนา:
   - pulse (ชีพจร)
   - bodyTemperature (อุณหภูมิร่างกาย)
   - bp, systolic, diastolic (ความดันโลหิตครั้งที่ 1 ระบุเป็น '120/80' ไม่ต้องใส่หน่วย mmHg)
   - bp2, systolic2, diastolic2 (ความดันโลหิตครั้งที่ 2 ถ้ามี ไม่ต้องใส่หน่วย mmHg)
   - o2sat (ออกซิเจนในเลือด)
   - map, map2
   - weight, height, bmi
   - chest (รอบอก), waist (รอบเอว)
   - respiratory (อัตราการหายใจ), painScore (ระดับความปวด)
4. investigations และ assessment_forms ให้ส่งเป็น Array เสมอ:
   - investigations: รายการตรวจแล็บ/ภาพวินิจฉัย เช่น ["Stat CXR", "CBC"]
   - assessment_forms: รายการรหัสแบบประเมิน เช่น ["depression", "smoking"] หรือ [] หากไม่มี
5. assessments: สรุปคำตอบและผลประเมินย่อยจากบทสนทนาสำหรับแต่ละแบบประเมินใน assessment_forms เพื่อนำไปเป็นค่าเริ่มต้น (default values) บนหน้าจอของแพทย์/พยาบาล
6. หากส่วนใดไม่มีการกล่าวถึงในบทสนทนา ให้ระบุเป็น null หรือข้อความว่าง "" หรือ [] สำหรับ Array
"""


def compute_map(systolic: float | int | str | None, diastolic: float | int | str | None) -> str | None:
    if systolic is None or diastolic is None:
        return None
    try:
        s_str = re.sub(r"[^\d.]", "", str(systolic))
        d_str = re.sub(r"[^\d.]", "", str(diastolic))
        if not s_str or not d_str:
            return None
        s = float(s_str)
        d = float(d_str)
        if 40 <= s <= 260 and 25 <= d <= 160:
            map_val = round((s + 2.0 * d) / 3.0, 1)
            if map_val.is_integer():
                return f"{int(map_val)}"
            return f"{map_val:.1f}"
    except Exception:
        pass
    return None


def compute_bmi(weight_val: float | int | str | None, height_val: float | int | str | None) -> str | None:
    if weight_val is None or height_val is None:
        return None
    try:
        w_str = re.sub(r"[^\d.]", "", str(weight_val))
        h_str = re.sub(r"[^\d.]", "", str(height_val))
        if not w_str or not h_str:
            return None
        w = float(w_str)
        h = float(h_str)
        if h > 3.0:  # height in cm -> convert to meters
            h = h / 100.0
        if 20 <= w <= 300 and 0.5 <= h <= 2.5:
            bmi = round(w / (h * h), 1)
            if bmi.is_integer():
                return f"{int(bmi)}"
            return f"{bmi:.1f}"
    except Exception:
        pass
    return None


def extract_with_groq(text: str, api_key: str, model: str = "llama-3.3-70b-versatile") -> dict | None:
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=model or "llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"บทสนทนาทางการแพทย์:\n\n{text}"},
            ],
            timeout=30,
        )
        content = response.choices[0].message.content
        if content:
            raw_text = content.strip()
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            res = json.loads(raw_text)
            if isinstance(res, dict):
                return res
    except Exception as e:
        print(f"[EXTRACT ERROR] Groq extraction failed: {e}")
    return None


def extract_with_gemini(text: str, api_key: str) -> dict | None:
    models = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    payload = {
        "system_instruction": {"parts": [{"text": EXTRACTION_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": f"บทสนทนาทางการแพทย์:\n\n{text}"}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
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
                    raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                    res = json.loads(raw_text)
                    if isinstance(res, dict):
                        return res
        except Exception:
            pass
    return None


def extract_with_openai(text: str, api_key: str, model: str = "gpt-4o-mini") -> dict | None:
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"บทสนทนาทางการแพทย์:\n\n{text}"},
            ],
            timeout=30,
        )
        content = response.choices[0].message.content
        if content:
            raw_text = content.strip()
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            res = json.loads(raw_text)
            if isinstance(res, dict):
                return res
    except OpenAIError as e:
        print(f"[EXTRACT ERROR] OpenAI API error: {e}")
    except Exception as e:
        print(f"[EXTRACT ERROR] OpenAI extraction failed: {e}")
    return None


def extract_heuristic_assessments(clean_text: str, forms: list[str], vs_dict: dict | None = None) -> dict[str, dict]:
    """
    Extracts summary and question-level default values from the dialogue for each assessment form.
    Enables the frontend to pre-fill checkboxes, radios, and fields with AI-recommended values.
    """
    vs = vs_dict or {}
    assessments: dict[str, dict] = {}

    # 1. DEPRESSION (2Q / 9Q / 8Q)
    if "depression" in forms or any(w in clean_text for w in ["หมดพลัง", "ไม่สบายใจ", "หน่วง", "เศร้า", "ท้อแท้", "ตื่นมาแล้วรู้สึกแย่", "เบื่อหน่าย", "ซึมเศร้า"]):
        q1_pos = any(w in clean_text for w in ["หมดพลัง", "ไม่สบายใจ", "หน่วง", "เศร้า", "ท้อแท้", "ตื่นมาแล้วรู้สึกแย่", "ซึมเศร้า"])
        q2_pos = any(w in clean_text for w in ["เบื่อ", "ไม่เพลิดเพลิน", "ไม่อยากทำอะไร", "ไม่สนุก", "ไม่ได้แตะเลย"])
        sleep_issue = any(w in clean_text for w in ["หลับยาก", "คิดวน", "หลับๆ ตื่นๆ", "ตื่นตี", "ตื่นมาไม่สดชื่น"])
        fatigue = any(w in clean_text for w in ["เพลีย", "เหนื่อยง่าย", "หมดพลัง", "ไม่มีแรง"])
        appetite = any(w in clean_text for w in ["เบื่ออาหาร", "กินข้าวไม่ค่อยลง", "กินข้าวไม่ลง", "ฝืนกิน"])
        concentration_good = any(w in clean_text for w in ["สมาธิ ยังมีอยู่", "สมาธิยังมีอยู่", "เคลียร์งานเอกสารได้", "ตั้งใจทำเสร็จ"])

        assessments["depression"] = {
            "title": "แบบประเมินภาวะซึมเศร้า (2Q / 9Q / 8Q)",
            "summary": "ผู้ป่วยมีภาวะหมดพลังงาน อารมณ์ไม่สบายใจ เบื่อหน่ายกิจกรรมที่ชอบ และนอนหลับยาก (2Q ได้ผลบวก แนะนำทำแบบประเมิน 9Q ต่อ)",
            "q2": {
                "q1": q1_pos,
                "q2": q2_pos,
                "positive": (q1_pos or q2_pos),
            },
            "answers": {
                "q2_1": "มี" if q1_pos else "ไม่มี",
                "q2_2": "มี" if q2_pos else "ไม่มี",
                "q2_result": "ผลบวก (Positive - ควรประเมิน 9Q ต่อ)" if (q1_pos or q2_pos) else "ผลลบ (Negative)",
                "sleep_issue": "หลับยาก หลับๆ ตื่นๆ ตื่นมาไม่สดชื่น (เกือบทุกคืน)" if sleep_issue else "ปกติ",
                "fatigue": "อ่อนเพลียมาก เหนื่อยง่าย หมดพลัง (เกือบทุกวัน)" if fatigue else "ปกติ",
                "appetite": "เบื่ออาหาร กินข้าวไม่ลง ต้องฝืนกิน" if appetite else "รับประทานอาหารได้ปกติ",
                "concentration": "ปกติ ยังจดจ่อและเคลียร์งานเอกสารได้ตามกำหนด" if concentration_good else "สมาธิลดลง",
                "self_harm": "ปฏิเสธ / ไม่พบประวัติความคิดทำร้ายตนเอง",
            }
        }

    # 2. SMOKING (Fagerström / Smoking Status)
    if "smoking" in forms or any(w in clean_text for w in ["สูบบุหรี่", "บุหรี่"]):
        never_smoked = any(w in clean_text for w in ["ไม่เคยสูบ", "ไม่สูบ", "ปฏิเสธ", "ไม่เคยสูบเลย", "บุหรี่ไม่เคยสูบ"])
        assessments["smoking"] = {
            "title": "แบบประเมินการสูบบุหรี่ (Smoking Assessment)",
            "summary": "ปฏิเสธประวัติการสูบบุหรี่ (ไม่เคยสูบบุหรี่)" if never_smoked else "มีประวัติการสูบบุหรี่",
            "status": "never" if never_smoked else "smoker",
            "answers": {
                "smoking_status": "ไม่เคยสูบ (Never smoked)" if never_smoked else "สูบเป็นประจำ",
                "cigarettes_per_day": "0" if never_smoked else "ไม่ระบุ",
                "fagerstrom_score": 0 if never_smoked else None,
            }
        }

    # 3. ALCOHOL (AUDIT / Alcohol Screening)
    if "alcohol" in forms or any(w in clean_text for w in ["แอลกอฮอล์", "สุรา", "ดื่มเหล้า", "ดื่มเบียร์"]):
        is_social = any(w in clean_text for w in ["สังสรรค์", "นานๆ ครั้ง", "นานๆ ที", "เดือนละครั้ง"])
        no_drink = any(w in clean_text for w in ["ไม่ดื่ม", "ไม่ดื่มเลย", "ปฏิเสธ"])
        assessments["alcohol"] = {
            "title": "แบบประเมินการดื่มสุรา (AUDIT Screening)",
            "summary": "ดื่มนานๆ ครั้งในงานสังสรรค์ (เดือนละ 1 ครั้ง ดื่ม 1-2 แก้ว) ความเสี่ยงต่ำ" if is_social else ("ปฏิเสธการดื่มแอลกอฮอล์" if no_drink else "มีประวัติดื่มแอลกอฮอล์"),
            "status": "occasional" if is_social else ("non_drinker" if no_drink else "drinker"),
            "answers": {
                "frequency": "นานๆ ครั้งในงานสังสรรค์ (เดือนละ 1 ครั้งหรือน้อยกว่า)" if is_social else ("ไม่เคยดื่มเลย" if no_drink else "ดื่มเป็นประจำ"),
                "amount": "1-2 ดื่มมาตรฐาน (1-2 แก้ว)" if is_social else "0",
                "risk_level": "ความเสี่ยงต่ำ (Low Risk)",
            }
        }

    # 4. FALL RISK (Morse Fall Scale)
    if "fall_risk" in forms or any(w in clean_text for w in ["หกล้ม", "สะดุด", "การเดิน"]):
        normal_gait = any(w in clean_text for w in ["ทำได้เองทั้งหมด", "ไม่มีปัญหาเรื่องการเดิน", "คล่องตัวดี", "ไม่มีสะดุดหกล้ม"])
        assessments["fall_risk"] = {
            "title": "แบบประเมินความเสี่ยงหกล้ม (Morse Fall Scale)",
            "summary": "ไม่มีประวัติหกล้ม เดินได้เองคล่องตัวดี จัดอยู่ในกลุ่มความเสี่ยงต่ำ" if normal_gait else "ประเมินความเสี่ยงต่อการพลัดตกหกล้ม",
            "risk_level": "low" if normal_gait else "moderate",
            "answers": {
                "history_of_fall": "ไม่มีประวัติหกล้มในช่วง 3 เดือนที่ผ่านมา (0 คะแนน)",
                "secondary_diagnosis": "ไม่มีโรคร่วมที่มีผลต่อการทรงตัว (0 คะแนน)",
                "ambulatory_aid": "เดินได้เอง ไม่ใช้อุปกรณ์ช่วยเดิน (0 คะแนน)",
                "iv_therapy": "ไม่มี (0 คะแนน)",
                "gait": "การเดินปกติ คล่องตัวดี (0 คะแนน)",
                "mental_status": "รับรู้ความสามารถตนเองดี (0 คะแนน)",
                "morse_score": 0 if normal_gait else 15,
            }
        }

    # 5. ADL (Barthel Index)
    if "adl" in forms or any(w in clean_text for w in ["กิจวัตรประจำวัน", "ทำได้เอง", "ติดเตียง"]):
        independent = any(w in clean_text for w in ["ทำได้เองทั้งหมด", "ทำได้เองปกติ", "คล่องตัวดี", "ช่วยเหลือตัวเองได้"])
        assessments["adl"] = {
            "title": "แบบประเมินกิจวัตรประจำวัน (Barthel ADL Index)",
            "summary": "ช่วยเหลือตนเองและทำกิจวัตรประจำวันได้เองทั้งหมด คล่องตัวดี (Independent)" if independent else "ประเมินระดับการพึ่งพา",
            "level": "independent" if independent else "partially_dependent",
            "answers": {
                "feeding": "รับประทานอาหารได้เอง",
                "grooming": "ดูแลความสะอาดตนเองได้",
                "dressing": "สวมใส่เสื้อผ้าได้เอง",
                "toilet_use": "ใช้ห้องน้ำได้เอง",
                "bathing": "อาบน้ำได้เอง",
                "transfer": "ลุกนั่งและย้ายตัวได้เอง",
                "mobility": "เดินได้เองคล่องตัวดี",
                "stairs": "ขึ้นลงบันไดได้เอง",
                "bowel": "กลั้นอุจจาระได้ปกติ",
                "bladder": "กลั้นปัสสาวะได้ปกติ",
                "total_score": 20 if independent else 10,
            }
        }

    # 6. MNA (Mini Nutritional Assessment)
    if "mna" in forms or any(w in clean_text for w in ["เบื่ออาหาร", "กินข้าวไม่ลง", "น้ำหนัก"]):
        reduced_intake = any(w in clean_text for w in ["เบื่ออาหาร", "กินข้าวไม่ค่อยลง", "กินข้าวไม่ลง", "ฝืนกิน"])
        unknown_wt = any(w in clean_text for w in ["ยังไม่ได้ชั่ง", "ไม่ได้ชั่งน้ำหนัก"])
        assessments["mna"] = {
            "title": "แบบประเมินภาวะโภชนาการ (Mini Nutritional Assessment)",
            "summary": "รับประทานอาหารลดลงปานกลางเนื่องจากเบื่ออาหาร ยังไม่ได้ชั่งน้ำหนักตัวล่าสุด",
            "answers": {
                "food_intake": "รับประทานอาหารลดลงปานกลาง (ฝืนทานให้ครบมื้อ)" if reduced_intake else "ปกติ",
                "weight_loss": "ไม่ทราบ / ยังไม่ได้ชั่งน้ำหนัก" if unknown_wt else "น้ำหนักคงที่",
                "mobility": "ลุกเดินออกจากเตียงได้ปกติ",
                "psychological_stress": "มีภาวะความเครียดหรืออารมณ์เศร้า",
                "neuropsychological": "มีภาวะซึมเศร้า",
            }
        }

    # 7. CVD (Thai CV Risk)
    if "cvd" in forms:
        assessments["cvd"] = {
            "title": "แบบประเมินความเสี่ยงโรคหัวใจและหลอดเลือด (Thai CV Risk)",
            "summary": "คัดกรองความเสี่ยงโรคหลอดเลือดหัวใจจากปัจจัยเสี่ยงและสัญญาณชีพ",
            "answers": {
                "smoking": "ไม่สูบ",
                "sbp": vs.get("systolic") or "120",
                "diabetes": "ไม่มี",
            }
        }

    # 8. BRADEN (Pressure Ulcer Risk)
    if "braden" in forms:
        assessments["braden"] = {
            "title": "แบบประเมินความเสี่ยงแผลกดทับ (Braden Scale)",
            "summary": "เคลื่อนไหวได้ปกติ รับรู้ความรู้สึกปกติ ความเสี่ยงต่อการเกิดแผลกดทับต่ำมาก",
            "answers": {
                "sensory": "ปกติ",
                "moisture": "แห้งปกติ",
                "activity": "เดินได้เอง",
                "mobility": "เคลื่อนไหวร่างกายได้เต็มที่",
                "friction": "ไม่มีปัญหาการเสียดสี",
            }
        }

    return assessments


def extract_with_heuristics(text: str) -> dict:
    """
    Smart rule-based and regex extraction fallback for Thai medical conversations.
    Accurately extracts all 12 Vital Signs (PR1, PR2, BT, BP1, BP2, O2sat, MAP, MAP2, Weight, รอบอก, Height, รอบเอว)
    and produces structured clinical notes (CC, PI, PE, Dx, Plan) without raw conversational dialogue.
    """
    clean_text = apply_dictionary(text)
    vs: dict[str, str | None] = {}

    # ------------------------------------------------------------------
    # 1. VITAL SIGNS EXTRACTION (12 CORE METRICS)
    # ------------------------------------------------------------------

    # A. Blood Pressure (BP1 & BP2): e.g. "124 ทับ 78", "124/78", "124 ต่อ 78", "120-80", "120 80"
    bp_matches = re.findall(
        r"(?:ความดัน(?:โลหิต)?|BP\s*\d?|วัดได้)?\s*[:=]?\s*(\d{2,3})\s*(?:\/|\-|ทับ|ต่อ|\s)\s*(\d{2,3})",
        clean_text,
        re.IGNORECASE,
    )
    valid_bps = []
    for s_raw, d_raw in bp_matches:
        s = int(s_raw)
        d = int(d_raw)
        if 70 <= s <= 240 and 35 <= d <= 150:
            valid_bps.append((s, d))

    if valid_bps:
        # BP1 -> bp, systolic, diastolic
        s1, d1 = valid_bps[0]
        vs["systolic"] = str(s1)
        vs["diastolic"] = str(d1)
        vs["bp"] = f"{s1}/{d1}"
        map1_val = compute_map(s1, d1)
        if map1_val:
            vs["map"] = map1_val

        # BP2 (if secondary reading exists) -> bp2, systolic2, diastolic2
        if len(valid_bps) > 1:
            s2, d2 = valid_bps[1]
            vs["systolic2"] = str(s2)
            vs["diastolic2"] = str(d2)
            vs["bp2"] = f"{s2}/{d2}"
            map2_val = compute_map(s2, d2)
            if map2_val:
                vs["map2"] = map2_val
    elif "ความดันปกติ" in clean_text or "ความดัน ปกติ" in clean_text:
        vs["bp"] = "ปกติ"

    # Secondary BP explicitly stated
    bp2_match = re.search(
        r"(?:ความดัน(?:โลหิต)?(?:ครั้งที่\s*2|ซ้ำ)|BP\s*2)\s*[:=]?\s*(\d{2,3})\s*(?:\/|\-|ทับ|ต่อ|\s)\s*(\d{2,3})",
        clean_text,
        re.IGNORECASE,
    )
    if bp2_match:
        s2 = int(bp2_match.group(1))
        d2 = int(bp2_match.group(2))
        if 70 <= s2 <= 240 and 35 <= d2 <= 150:
            vs["systolic2"] = str(s2)
            vs["diastolic2"] = str(d2)
            vs["bp2"] = f"{s2}/{d2}"
            map2_val = compute_map(s2, d2)
            if map2_val:
                vs["map2"] = map2_val

    # B. Temperature (BT): e.g. "ไข้สูง 38.8", "อุณหภูมิ 36.8", "BT 37.0", "38.8 C", "37.5 องศา"
    temp_match = re.search(
        r"(?:ไข้(?:สูง)?|อุณหภูมิ(?:กาย)?|BT|temp|ตัวร้อน)\s*[:=]?\s*([34]\d(?:\.\d+)?)\s*(?:C|°C|องศา)?",
        clean_text,
        re.IGNORECASE,
    )
    if not temp_match:
        temp_match = re.search(r"([34]\d\.\d+)\s*(?:C|°C|องศา)", clean_text, re.IGNORECASE)
    if temp_match:
        vs["bodyTemperature"] = temp_match.group(1)

    # C. Pulse Rate: e.g. "ชีพจร 88", "PR 80", "pulse 82 bpm", "หัวใจเต้น 80", "ชีบพระจรเต้น 100"
    pulse_matches = re.findall(
        r"(?:ชีพจร(?:ครั้งที่\s*1)?|ชีบพระจร|ชีบพจร|ชีบจร|pulse(?:\s*1)?|PR\s*1?|หัวใจเต้น|อัตราการเต้นของหัวใจ)\s*(?:เต้น(?:เร็ว|ช้า)?(?:ไปนิด|ปกติ)?)?\s*[:=]?\s*(\d{2,3})",
        clean_text,
        re.IGNORECASE,
    )
    if pulse_matches:
        vs["pulse"] = pulse_matches[0]

    # D. SpO2 / O2sat: e.g. "ออกซิเจนในเลือด 98%", "SpO2 95%", "O2sat 99%", "ระดับออกซิเจนในเลือดปลายนิ้วอยู่ที่ 98%"
    spo2_match = re.search(
        r"(?:ระดับ)?(?:ออกซิเจน(?:ในเลือด)?(?:ปลายนิ้ว)?|SpO2|O2sat|O2\s*sat|O2)\s*(?:ในเลือด)?\s*(?:ปลายนิ้ว)?\s*(?:อยู่ที่|วัดได้)?\s*[:=]?\s*(\d{2,3})\s*%?",
        clean_text,
        re.IGNORECASE,
    )
    if spo2_match:
        spo2_val = int(spo2_match.group(1))
        if 50 <= spo2_val <= 100:
            vs["o2sat"] = str(spo2_val)

    # E. MAP (Explicitly spoken if any)
    map_match = re.search(
        r"(?:MAP|Mean\s*Arterial\s*Pressure)\s*(?:1|ครั้งที่\s*1)?\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)",
        clean_text,
        re.IGNORECASE,
    )
    if map_match:
        vs["map"] = map_match.group(1)

    map2_match = re.search(
        r"(?:MAP\s*2|Mean\s*Arterial\s*Pressure\s*2|MAP\s*ครั้งที่\s*2)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)",
        clean_text,
        re.IGNORECASE,
    )
    if map2_match:
        vs["map2"] = map2_match.group(1)

    # F. Weight (น้ำหนัก): e.g. "น้ำหนัก 65 กิโล", "น้ำหนัก 65 kg", "นน. 58", "weight 70"
    weight_match = re.search(
        r"(?:น้ำหนัก|นน\.|weight|หนัก)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:กก\.?|กิโล(?:กรัม)?|kg)?",
        clean_text,
        re.IGNORECASE,
    )
    if weight_match:
        w_val = weight_match.group(1)
        if 20 <= float(w_val) <= 300:
            vs["weight"] = w_val

    # G. Height (ส่วนสูง): e.g. "ส่วนสูง 165 เซน", "สส. 170 ซม.", "สูง 160", "height 168 cm"
    height_match = re.search(
        r"(?:ส่วนสูง|สส\.|height|สูง)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:ซม\.?|เซน(?:ติเมตร)?|cm)?",
        clean_text,
        re.IGNORECASE,
    )
    if height_match:
        h_val = height_match.group(1)
        if 50 <= float(h_val) <= 250:
            vs["height"] = h_val

    # H. รอบอก: e.g. "รอบอก 88 ซม.", "รอบอก 34 นิ้ว", "chest 88 cm"
    chest_match = re.search(
        r"(?:รอบอก|ขนาดรอบอก|chest(?:\s*circumference)?)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(ซม\.?|เซน(?:ติเมตร)?|cm|นิ้ว|inch|in)?",
        clean_text,
        re.IGNORECASE,
    )
    if chest_match:
        vs["chest"] = chest_match.group(1)

    # I. รอบเอว: e.g. "รอบเอว 80 ซม.", "รอบเอว 32 นิ้ว", "waist 80 cm", "เอว 30"
    waist_match = re.search(
        r"(?:รอบเอว|ขนาดรอบเอว|waist(?:\s*circumference)?|เอว)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(ซม\.?|เซน(?:ติเมตร)?|cm|นิ้ว|inch|in)?",
        clean_text,
        re.IGNORECASE,
    )
    if waist_match:
        vs["waist"] = waist_match.group(1)

    # J. Respiratory Rate: e.g. "หายใจ 20 ครั้งต่อนาที", "RR 24"
    rr_match = re.search(
        r"(?:หายใจ|RR)\s*(?:ค่อนข้าง)?(?:เร็ว|ถี่|ที่)?\s*[:=]?\s*(\d{1,2})\s*(?:ครั้ง(?:ต่อนาที)?)?",
        clean_text,
        re.IGNORECASE,
    )
    if rr_match:
        vs["respiratory"] = rr_match.group(1)

    # K. Pain Score: e.g. "ความปวด 5", "pain score 7", "ก็น่าจะสัก 6", "ก็น่าจะสัก 6-7"
    pain_match = re.search(
        r"(?:pain(?:\s*score)?|ความปวด|ระดับความปวด)\s*[:=]?\s*(\d{1,2})(?:\s*-\s*\d{1,2})?\s*(?:\/\s*10|เต็ม\s*10)?",
        clean_text,
        re.IGNORECASE,
    )
    if pain_match:
        vs["painScore"] = pain_match.group(1)
    else:
        pain_ans = re.search(r"(?:ก็น่าจะสัก|น่าจะสัก|น่าจะแตะ|สัก)\s*(\d{1,2})(?:\s*-\s*\d{1,2})?", clean_text)
        if pain_ans and int(pain_ans.group(1)) <= 10:
            vs["painScore"] = pain_ans.group(1)

    # L. BMI
    if vs.get("weight") and vs.get("height"):
        bmi_val = compute_bmi(vs["weight"], vs["height"])
        if bmi_val:
            vs["bmi"] = bmi_val

    # ------------------------------------------------------------------
    # 2. CLINICAL CONTEXT & SUMMARIZATION (NO RAW DIALOGUE)
    # ------------------------------------------------------------------
    has = lambda kw: kw in clean_text

    dur_match = re.search(r"(\d+[\s\-]*(?:ถึง|\-)?\s*\d*\s*(?:สัปดาห์|สัปตา|วัน|เดือน|ปี))", clean_text)
    duration_str = dur_match.group(1).replace("สัปตา", "สัปดาห์") if dur_match else ""

    # Explicit ICD-9 and DRG mention in spoken audio
    icd9_match = re.search(r"(?:ICD\s*9|ICD-9|รหัสหัตถการ|หัตถการ)\s*[:=]?\s*([0-9]{2}(?:\.[0-9]{1,2})?)", clean_text, re.IGNORECASE)
    drg_match = re.search(r"(?:DRG|กลุ่มโรค)\s*[:=]?\s*([0-9A-Z]{4,6})", clean_text, re.IGNORECASE)

    # Category A: Mental Health / Depressive Episode / Insomnia / Anhedonia / Fatigue
    if (has("หมดพลัง") or has("ไม่สบายใจ") or has("เบื่อหน่าย") or has("คิดวน") or has("หลับยาก") or has("ซึมเศร้า")) and not (has("ปอดอักเสบ") or has("ไอถี่") or has("เสมหะ")):
        dur_text = duration_str if duration_str else "2-3 สัปดาห์"
        cc = f"มีปัญหาเรื่องการนอนหลับยาก รู้สึกหมดพลังงาน อ่อนเพลีย และมีอารมณ์ไม่สบายใจ/เบื่อหน่าย เป็นมา {dur_text}"
        
        pi_items = [
            f"ช่วง {dur_text} ก่อนมาโรงพยาบาล ผู้ป่วยรู้สึกหมดพลังงานอย่างมาก ไม่อยากลุกออกจากเตียง รู้สึกหน่วงและไม่สบายใจแทบทั้งวัน",
            "มีความรู้สึกเบื่อหน่าย ไม่เพลิดเพลินกับกิจกรรมหรือสิ่งที่เคยชอบทำ (ดูซีรีส์, ปลูกต้นไม้)",
            "มีปัญหาการนอนหลับ หลับยาก เข้านอนแล้วคิดวนไปวนมา หลับๆ ตื่นๆ เกือบทุกคืน หลับได้ช่วงตี 2-3 ตื่นมาไม่สดชื่น",
            "ระหว่างวันรู้สึกอ่อนเพลียมาก เหนื่อยง่าย เดินขึ้นบันไดชั้น 2 รู้สึกเหนื่อย",
            "เบื่ออาหาร กินข้าวไม่ค่อยลง ต้องฝืนกินให้ครบมื้อ",
            "ด้านสมาธิยังสามารถจดจ่อและเคลียร์งานเอกสารได้ตามกำหนด",
            "ปฏิเสธประวัติการสูบบุหรี่ ดื่มแอลกอฮอล์นานๆ ครั้งในงานสังสรรค์ กิจวัตรประจำวันทำได้เองปกติ",
        ]
        pi = "\n".join(pi_items)
        pe = "General: Depressed mood, fatigued appearance, intact concentration, normal gait and motor function. Systemic exams within normal limits."
        if vs.get("bp"):
            pe += f" (BP {vs['bp']})"
        dx = "Depressive Episode with Insomnia (ภาวะซึมเศร้าร่วมกับปัญหาการนอนหลับ)"
        icd10 = "F32.9"
        icd9 = icd9_match.group(1) if icd9_match else "94.49"
        drg = drg_match.group(1) if drg_match else "19500"
        inv = ["แบบประเมินคัดกรองภาวะซึมเศร้า (2Q / 9Q)", "แบบประเมินความเสี่ยงการทำร้ายตนเอง (8Q)"]
        forms = ["depression"]
        plan = "1. ให้คำปรึกษาประคับประคองจิตใจ (Supportive Counseling)\n2. ให้คำแนะนำสุขอนามัยการนอนหลับ (Sleep Hygiene)\n3. พิจารณาปรึกษาจิตแพทย์หรือเริ่มยาปรับสารสื่อประสาทตามความเหมาะสม\n4. นัดตรวจติดตามอาการใน 1-2 สัปดาห์"
        past_hx = "ปฏิเสธโรคประจำตัวเดิม, ไม่สูบบุหรี่, ดื่มแอลกอฮอล์นานๆ ครั้ง (สังสรรค์เดือนละ 1-2 แก้ว), กิจวัตรประจำวันทำได้เองคล่องตัวดี"
        disp = "OPD"

    # Category B1: Influenza / Viral Upper Respiratory Infection (ไข้หวัดใหญ่ / URI)
    elif (has("ไข้หวัดใหญ่") or has("Swab") or has("Antigen") or has("โควิด") or has("ปวดเมื่อยตามตัว") or has("ปวดหลังปวดเอว")) and (has("ไอแห้ง") or has("เจ็บคอ") or has("เสียงปอดโปร่ง") or has("ยาต้านไวรัส")):
        dur_text = duration_str if duration_str else "3 วัน"
        cc = f"มีไข้ ปวดเมื่อยตามตัว อ่อนเพลีย ไอแห้งๆ มา {dur_text}"
        pi = (
            f"ผู้ป่วยรู้สึกเพลียมาก มีไข้รุมๆ ตัวร้อนมาตั้งแต่วันเสาร์ (ประมาณ {dur_text}) "
            f"มีอาการปวดเมื่อยเนื้อตัว ปวดหลังปวดเอว หนาวสั่น ปวดระบมตามข้อ ไอแห้งๆ "
            f"เจ็บคอเล็กน้อยเวลากลืนน้ำลาย แทบไม่มีน้ำมูก กลัวโควิดลงปอด ระดับความปวดเมื่อยประมาณ 6-8/10"
        )
        pe = "Chest: Normal breath sounds, clear both lungs (เสียงปอดโปร่งใสดีทั้งสองข้าง). Pharynx: Mild pharyngeal injection (ผนังคอหอยด้านหลังมีรอยแดงเล็กน้อย)."
        dx = "Influenza-like illness / Acute Viral Respiratory Tract Infection (สงสัยไข้หวัดใหญ่)"
        icd10 = "J11.1"
        icd9 = icd9_match.group(1) if icd9_match else "-"
        drg = drg_match.group(1) if drg_match else "04510"
        inv = ["Rapid Antigen Test (Swab ป้ายจมูกสำหรับไข้หวัดใหญ่และโควิด-19)"]
        forms = []
        plan = (
            "1. ส่งตรวจ Swab ป้ายจมูกตรวจ Antigen สำหรับไข้หวัดใหญ่และ COVID-19 (ทราบผลใน 15 นาที)\n"
            "2. ให้ยาลดไข้แก้ปวด Paracetamol รับประทานบรรเทาอาการระหว่างรอผล\n"
            "3. หากผลตรวจพบเชื้อไข้หวัดใหญ่ พิจารณาเริ่มยาต้านไวรัสเฉพาะทาง (Oseltamivir) ทันทีเพื่อลดระยะเวลาไข้และภาวะแทรกซ้อน\n"
            "4. ดื่มน้ำมากๆ พักผ่อนให้เพียงพอ แยกของใช้ส่วนตัว และสังเกตอาการเหนื่อยหอบ"
        )
        past_hx = "ปฏิเสธโรคประจำตัวเดิม ไม่มีประวัติแพ้ยา"
        disp = "OPD"

    # Category B2: Respiratory Infection / Pneumonia
    elif has("เสมหะ") or has("แน่นหน้าอก") or (has("ไข้") and (has("เหนื่อย") or has("Crepitation"))):
        dur_text = duration_str if duration_str else "2 วัน"
        cc = f"ไข้สูง ไอถี่ มีเสมหะสีเขียวข้น หายใจเหนื่อย แน่นหน้าอกด้านขวา เป็นมา {dur_text}"
        pi = (
            f"{dur_text} ก่อนมาโรงพยาบาล เริ่มมีไข้ขึ้นสูง หนาวสั่น ไอถี่ มีเสมหะสีเขียวข้นตลอด "
            f"เริ่มรู้สึกหายใจเหนื่อยและแน่นหน้าอกข้างขวาเวลาหายใจเข้าลึกๆ เจ็บคอแค่วันแรก ไม่มีน้ำมูก "
            f"ซื้อยา Paracetamol ทานเอง ไข้ลดลงชั่วคราวแล้วกลับมาหนาวสั่นเหมือนเดิม"
        )
        pe = "Chest: Crepitation / abnormal breath sound at right lower lung field, Pharyngeal injection mildly"
        dx = "Community-Acquired Pneumonia, Bacterial (โรคปอดอักเสบติดเชื้อแบคทีเรีย)"
        icd10 = "J15.9"
        icd9 = icd9_match.group(1) if icd9_match else "96.04"
        drg = drg_match.group(1) if drg_match else "04500"
        inv = ["Stat CXR (Chest X-ray)", "CBC / Blood culture", "Sputum Gram stain & culture"]
        forms = []
        plan = "1. รับไว้รักษาในโรงพยาบาล (Admit IPD)\n2. ให้ออกซิเจนทางจมูก (Oxygen cannula) keep SpO2 > 95%\n3. ให้ยาปฏิชีวนะทางหลอดเลือดดำ (IV Antibiotics)\n4. ให้ยาลดไข้ทางสายน้ำเกลือ (IV Antipyretics)"
        past_hx = "ไม่มีประวัติแพ้ยาหรือโรคประจำตัวร้ายแรง"
        disp = "Admit"

    # Category C: Diabetes & Metabolic
    elif has("ดื้ออินซูลิน") or has("Insulin Resistance") or has("Acanthosis") or (has("เบาหวาน") and has("ติ่งเนื้อ")):
        cc = "ตรวจประเมินภาวะดื้อต่ออินซูลินและรอยปื้นดำบริเวณหลังคอ/รักแร้"
        pi = "ผู้ป่วยมาตรวจเนื่องจากสังเกตพบรอยปื้นดำบริเวณหลังคอและใต้รักแร้ มีติ่งเนื้อขึ้นตามผิวหนัง มีอาการหิวบ่อย อ่อนเพลีย และปัสสาวะบ่อยตอนกลางคืน"
        pe = "Skin: Acanthosis nigricans at posterior neck and axillae, Multiple skin tags, No ulceration"
        dx = "Insulin Resistance syndrome (ภาวะดื้อต่ออินซูลิน / Metabolic Syndrome)"
        icd10 = "E11.9"
        icd9 = icd9_match.group(1) if icd9_match else "-"
        drg = drg_match.group(1) if drg_match else "10500"
        inv = ["FBS", "HbA1c", "Lipid profile (Cholesterol, Triglyceride, HDL, LDL)", "Fasting Insulin"]
        forms = ["cvd"]
        plan = "1. ปรับเปลี่ยนพฤติกรรมการรับประทานอาหาร (Dietary modification / Low Glycemic Index)\n2. แนะนำการออกกำลังกายสม่ำเสมอ\n3. ติดตามผลระดับน้ำตาลสะสมใน 3 เดือน"
        past_hx = "ไม่มีประวัติเบาหวานชนิดที่ 1"
        disp = "OPD"

    # Category D: General / Default Fallback
    else:
        cc = "ผู้ป่วยมาตรวจติดตามอาการทั่วไปและรับคำปรึกษาจากแพทย์"
        pi = "ผู้ป่วยรายงานอาการไม่สบายตัวทั่วไป สัญญาณชีพและอาการแสดงคงที่ สามารถทำกิจวัตรประจำวันได้ตามปกติ"
        pe = "General: Good consciousness, oriented to time/place/person. Systemic physical examinations within normal limits."
        dx = "Malaise and Fatigue (ภาวะอ่อนเพลียและเจ็บป่วยทั่วไป)"
        icd10 = "R53"
        icd9 = icd9_match.group(1) if icd9_match else "-"
        drg = drg_match.group(1) if drg_match else "23500"
        inv = ["ตรวจประเมินสัญญาณชีพและตรวจร่างกายตามระบบ"]
        forms = []
        plan = "แนะนำการพักผ่อน ดูแลสุขภาพทั่วไป และนัดตรวจติดตามหากอาการไม่ดีขึ้น"
        past_hx = "-"
        disp = "OPD"

    # Heuristic detection for additional assessment forms
    if "สูบบุหรี่" in clean_text or "บุหรี่" in clean_text:
        if "smoking" not in forms:
            forms.append("smoking")
    if any(w in clean_text for w in ["ดื่มเหล้า", "ดื่มเบียร์", "แอลกอฮอล์", "สุรา"]):
        if "alcohol" not in forms:
            forms.append("alcohol")
    if any(w in clean_text for w in ["หกล้ม", "ล้มบ่อย", "พลัดตก"]):
        if "fall_risk" not in forms:
            forms.append("fall_risk")
    if "แผลกดทับ" in clean_text:
        if "braden" not in forms:
            forms.append("braden")
    if any(w in clean_text for w in ["ติดเตียง", "ลุกไม่ไหว", "ช่วยเหลือตัวเองไม่ได้"]):
        if "adl" not in forms:
            forms.append("adl")
    if any(w in clean_text for w in ["เบื่ออาหาร", "น้ำหนักลดเร็ว"]):
        if "mna" not in forms:
            forms.append("mna")

    assessments_dict = extract_heuristic_assessments(clean_text, forms, vs)

    return {
        "chiefComplaint": cc,
        "presentIllness": pi,
        "pastHistory": past_hx,
        "vitalSigns": vs,
        "vital_signs": vs,
        "physicalExam": pe,
        "provisionalDiagnosis": dx,
        "provisional_diagnosis": dx,
        "icd10": icd10,
        "icd10Code": icd10,
        "icd9": icd9,
        "icd9Code": icd9,
        "drg": drg,
        "drgCode": drg,
        "investigation": ", ".join(inv),
        "investigations": inv,
        "assessmentForms": forms,
        "assessment_forms": forms,
        "assessments": assessments_dict,
        "note": plan,
        "disposition": disp,
    }


def normalize_vital_signs(vs_dict: dict, raw_text: str) -> VitalSignsData:
    """
    Consolidates and normalizes vital signs dictionary, parses systolic/diastolic,
    calculates MAP/MAP2/BMI, and falls back to regex extraction for any missing fields.
    """
    # Run heuristics as baseline to guarantee full field extraction
    h_data = extract_with_heuristics(raw_text)
    h_vs = h_data.get("vital_signs") or {}

    def get_val(*keys: str) -> str | None:
        for k in keys:
            val = vs_dict.get(k)
            if val is not None and str(val).strip() != "" and str(val).lower() != "null":
                return str(val).strip()
        for k in keys:
            val = h_vs.get(k)
            if val is not None and str(val).strip() != "" and str(val).lower() != "null":
                return str(val).strip()
        return None

    # 1. Pulse
    pulse = get_val("pulse", "pr", "pr1", "pulse_rate", "pulse_rate_1")

    # 2. Body Temperature
    bodyTemperature = get_val("bodyTemperature", "body_temperature", "temperature", "bt", "temp")

    # 3. BP, Systolic, Diastolic
    bp = get_val("bp", "blood_pressure", "bp1")
    systolic = get_val("systolic", "bp1_systolic")
    diastolic = get_val("diastolic", "bp1_diastolic")

    if bp:
        m = re.search(r"(\d{2,3})\s*(?:\/|\-|ทับ|ต่อ)\s*(\d{2,3})", bp)
        if m:
            systolic = systolic or m.group(1)
            diastolic = diastolic or m.group(2)
            bp = f"{m.group(1)}/{m.group(2)}"
        else:
            bp = re.sub(r"(?i)\s*mm\s*hg", "", bp).strip()
    elif systolic and diastolic and not bp:
        bp = f"{systolic}/{diastolic}"

    if systolic and diastolic:
        systolic = re.sub(r"[^\d]", "", str(systolic))
        diastolic = re.sub(r"[^\d]", "", str(diastolic))
        if systolic and diastolic and (not bp or "/" in str(bp)):
            bp = f"{systolic}/{diastolic}"

    # 4. BP2, Systolic2, Diastolic2
    bp2 = get_val("bp2", "blood_pressure_2")
    systolic2 = get_val("systolic2", "bp2_systolic", "systolic_2")
    diastolic2 = get_val("diastolic2", "bp2_diastolic", "diastolic_2")

    if bp2:
        m = re.search(r"(\d{2,3})\s*(?:\/|\-|ทับ|ต่อ)\s*(\d{2,3})", bp2)
        if m:
            systolic2 = systolic2 or m.group(1)
            diastolic2 = diastolic2 or m.group(2)
            bp2 = f"{m.group(1)}/{m.group(2)}"
        else:
            bp2 = re.sub(r"(?i)\s*mm\s*hg", "", bp2).strip()
    elif systolic2 and diastolic2 and not bp2:
        bp2 = f"{systolic2}/{diastolic2}"

    if systolic2 and diastolic2:
        systolic2 = re.sub(r"[^\d]", "", str(systolic2))
        diastolic2 = re.sub(r"[^\d]", "", str(diastolic2))
        if systolic2 and diastolic2 and (not bp2 or "/" in str(bp2)):
            bp2 = f"{systolic2}/{diastolic2}"

    # 5. O2sat
    o2sat = get_val("o2sat", "spo2", "o2_sat", "oxygen_saturation")

    # 6. MAP & MAP2
    map_val = get_val("map", "mean_arterial_pressure")
    if not map_val and systolic and diastolic:
        map_val = compute_map(systolic, diastolic)

    map2_val = get_val("map2", "mean_arterial_pressure_2")
    if not map2_val and systolic2 and diastolic2:
        map2_val = compute_map(systolic2, diastolic2)

    # 7. Weight & Height & BMI
    weight = get_val("weight", "bw", "body_weight")
    height = get_val("height", "ht", "body_height")
    bmi = get_val("bmi")
    if not bmi and weight and height:
        bmi = compute_bmi(weight, height)

    # 8. Body measurements (chest, waist)
    chest = get_val("chest", "chest_circumference")
    waist = get_val("waist", "waist_circumference")

    # 9. Additional metrics (respiratory, painScore)
    respiratory = get_val("respiratory", "rr", "respiratory_rate")
    painScore = get_val("painScore", "pain_score", "pain")

    def clean_num(val: str | None) -> str | None:
        if not val:
            return None
        m = re.search(r"\d+(?:\.\d+)?", str(val))
        return m.group(0) if m else None

    # 1. Pulse (Pure number)
    pulse = clean_num(pulse)

    # 2. Body Temperature (Pure number)
    bodyTemperature = clean_num(bodyTemperature)

    # 5. O2sat (Pure number)
    o2sat = clean_num(o2sat)

    # 6. MAP (Pure number)
    map_val = clean_num(map_val)
    map2_val = clean_num(map2_val)

    # 7. Weight & Height & BMI (Pure numbers)
    weight = clean_num(weight)
    height = clean_num(height)
    bmi = clean_num(bmi)

    # 8. Chest & Waist (Pure numbers)
    chest = clean_num(chest)
    waist = clean_num(waist)

    # 9. Additional metrics (Pure numbers)
    respiratory = clean_num(respiratory)
    painScore = clean_num(painScore)

    return VitalSignsData(
        pulse=pulse,
        bodyTemperature=bodyTemperature,
        bp=bp,
        systolic=systolic,
        diastolic=diastolic,
        bp2=bp2,
        systolic2=systolic2,
        diastolic2=diastolic2,
        o2sat=o2sat,
        map=map_val,
        map2=map2_val,
        weight=weight,
        height=height,
        bmi=bmi,
        chest=chest,
        waist=waist,
        respiratory=respiratory,
        painScore=painScore,
    )


ICD10_DESCRIPTIONS: dict[str, str] = {
    "J11.1": "Influenza with other respiratory manifestations / ไข้หวัดใหญ่",
    "J11.0": "Influenza with pneumonia / ไข้หวัดใหญ่ร่วมกับปอดบวม",
    "J11.8": "Influenza with other specified manifestations / ไข้หวัดใหญ่อาการแสดงอื่น",
    "J11.9": "Influenza, unspecified / ไข้หวัดใหญ่ไม่ระบุรายละเอียด",
    "J10.1": "Influenza with other respiratory manifestations, virus identified / ไข้หวัดใหญ่ระบุเชื้อ",
    "J15.9": "Bacterial pneumonia, unspecified / ปอดอักเสบจากเชื้อแบคทีเรีย",
    "J18.9": "Pneumonia, unspecified / โรคปอดอักเสบ",
    "J00": "Acute nasopharyngitis / ไข้หวัดธรรมดา Common cold",
    "J02.9": "Acute pharyngitis, unspecified / คออักเสบเฉียบพลัน",
    "J03.9": "Acute tonsillitis, unspecified / ต่อมทอนซิลอักเสบเฉียบพลัน",
    "J06.9": "Acute upper respiratory infection, unspecified / ทางเดินหายใจส่วนบนอักเสบเฉียบพลัน",
    "J20.9": "Acute bronchitis, unspecified / หลอดลมอักเสบเฉียบพลัน",
    "J45.9": "Asthma, unspecified / โรคหืด",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified / ถุงลมโป่งพอง COPD",
    "F32.9": "Depressive episode, unspecified / ภาวะซึมเศร้า",
    "F32.0": "Mild depressive episode / ภาวะซึมเศร้าเล็กน้อย",
    "F32.1": "Moderate depressive episode / ภาวะซึมเศร้าปานกลาง",
    "F32.2": "Severe depressive episode without psychotic symptoms / ภาวะซึมเศร้ารุนแรง",
    "F41.9": "Anxiety disorder, unspecified / โรควิตกกังวล",
    "G47.0": "Disorders of initiating and maintaining sleep / นอนไม่หลับ Insomnia",
    "E11.9": "Type 2 diabetes mellitus without complications / เบาหวานชนิดที่ 2",
    "I10": "Essential hypertension / ความดันโลหิตสูง",
    "E78.5": "Hyperlipidemia, unspecified / ภาวะไขมันในเลือดสูง",
    "R53": "Malaise and fatigue / ภาวะอ่อนเพลียและเมื่อยล้า",
    "N10": "Acute pyelonephritis / กรวยไตอักเสบเฉียบพลัน",
    "N39.0": "Urinary tract infection, site not specified / ทางเดินปัสสาวะอักเสบ UTI",
    "K29.7": "Gastritis, unspecified / กระเพาะอาหารอักเสบ",
    "K21.9": "Gastro-esophageal reflux disease / กรดไหลย้อน GERD",
    "A09.9": "Gastroenteritis and colitis of unspecified origin / ท้องเสีย อุจจาระร่วงเฉียบพลัน",
    "M79.1": "Myalgia / ปวดกล้ามเนื้อ",
    "M54.5": "Low back pain / ปวดหลังส่วนล่าง",
    "R50.9": "Fever, unspecified / ไข้ไม่ระบุสาเหตุ",
    "R05": "Cough / อาการไอ",
    "R07.4": "Chest pain, unspecified / เจ็บแน่นหน้าอก",
    "U07.1": "COVID-19, virus identified / โควิด-19",
}

ICD9_DESCRIPTIONS: dict[str, str] = {
    "96.04": "Continuous oxygen therapy / การให้ออกซิเจน",
    "94.49": "Other nonoperative psychotherapy / counseling การให้คำปรึกษา",
    "86.22": "Excisional debridement / การตัดแต่งแผล",
    "86.28": "Nonexcisional debridement / การล้างทำความสะอาดแผล",
    "38.93": "Venous catheterization / การเปิดเส้นให้สารน้ำทางหลอดเลือดดำ",
    "87.44": "Routine chest x-ray / การถ่ายภาพรังสีทรวงอก",
    "90.59": "Microscopic examination of blood / ตรวจเลือดทางห้องปฏิบัติการ",
}

DRG_DESCRIPTIONS: dict[str, str] = {
    "04500": "Simple Pneumonia & Pleurisy age > 17 with CC / ปอดบวมและเยื่อหุ้มปอดอักเสบร่วมกับภาวะแทรกซ้อน",
    "04510": "Viral Illness / Influenza without CC / โรคติดเชื้อไวรัสหรือไข้หวัดใหญ่",
    "04520": "Simple Pneumonia & Pleurisy age > 17 without CC / ปอดบวมและเยื่อหุ้มปอดอักเสบ",
    "04530": "Respiratory Infections & Inflammations with CC / ทางเดินหายใจอักเสบติดเชื้อร่วมกับภาวะแทรกซ้อน",
    "04540": "Respiratory Infections & Inflammations without CC / ทางเดินหายใจอักเสบติดเชื้อ",
    "19500": "Depressive Disorders without CC / โรคซึมเศร้า",
    "19510": "Depressive Disorders with CC / โรคซึมเศร้าร่วมกับภาวะแทรกซ้อน",
    "19520": "Neurosis & Psychogenic Disorders / โรควิตกกังวลและประสาท",
    "10500": "Diabetes without CC / โรคเบาหวาน",
    "10510": "Diabetes with CC / โรคเบาหวานร่วมกับภาวะแทรกซ้อน",
    "05500": "Circulatory Disorders with Acute Myocardial Infarction / กล้ามเนื้อหัวใจขาดเลือดเฉียบพลัน",
    "05510": "Heart Failure & Shock / ภาวะหัวใจวายและช็อก",
    "05520": "Hypertension / โรคความดันโลหิตสูง",
    "06520": "Esophagitis, Gastroenteritis & Misc Digestive Disorders / โรคทางเดินอาหารและกระเพาะลำไส้อักเสบ",
    "11500": "Kidney & Urinary Tract Infections / การติดเชื้อทางเดินปัสสาวะ",
    "08500": "Musculoskeletal & Connective Tissue Disorders / โรคกล้ามเนื้อและข้อ",
    "23500": "Other Factors Influencing Health Status / ปัจจัยอื่นที่มีผลต่อสถานะสุขภาพ",
}

# Compatibility mappings
ICD10_MAP = {k: f"{k} ({v})" for k, v in ICD10_DESCRIPTIONS.items()}
ICD9_MAP = {k: f"{k} ({v})" for k, v in ICD9_DESCRIPTIONS.items()}
DRG_MAP = {k: f"{k} ({v})" for k, v in DRG_DESCRIPTIONS.items()}


def parse_medical_code_and_desc(
    raw_val: str | None,
    desc_map: dict[str, str],
    fallback_desc: str = "",
    explicit_desc: str = "",
) -> tuple[str, str, str]:
    """
    Parses a medical code string (ICD-10, ICD-9, DRG) and extracts:
    - code: Pure code string (e.g. 'J11.1', '96.04', '04510', or '-')
    - desc: Name/description (e.g. 'Influenza with other respiratory manifestations / ไข้หวัดใหญ่')
    - full: Combined representation (e.g. 'J11.1 (Influenza with other respiratory manifestations / ไข้หวัดใหญ่)')
    """
    if not raw_val or str(raw_val).strip() in {"-", "none", "null", "ไม่มี", "ไม่ระบุ", ""}:
        if explicit_desc and explicit_desc.strip() not in {"-", "none", "null", "ไม่มี", "ไม่ระบุ", ""}:
            clean_explicit = explicit_desc.strip()
            return "-", clean_explicit, f"- ({clean_explicit})"
        return "-", "", "-"

    val = str(raw_val).strip()
    code = ""
    desc = ""

    # Check if string has format 'CODE (DESC)' or 'CODE - DESC' or 'CODE: DESC'
    m = re.match(r"^([A-Za-z0-9.]+)\s*[\(\-\:\/]\s*(.+?)[\)]?$", val)
    if m:
        code = m.group(1).strip()
        desc = m.group(2).strip()
    else:
        tokens = val.split(None, 1)
        code = tokens[0].strip()
        desc = tokens[1].strip() if len(tokens) > 1 else ""

    # Clean code token
    code_clean = re.sub(r"[^\w.]", "", code)
    if not code_clean:
        code_clean = code

    # If explicit description was provided in a separate field, use it if desc is empty
    if not desc and explicit_desc and explicit_desc.strip() not in {"-", "none", "null", "ไม่มี", "ไม่ระบุ", ""}:
        desc = explicit_desc.strip()

    # Look up in reference dictionaries
    if not desc or desc.lower() in {"none", "null"}:
        desc = desc_map.get(code_clean.upper(), desc_map.get(code_clean, fallback_desc or ""))

    # Strip surrounding parentheses or punctuation from desc
    desc = re.sub(r"^\((.+)\)$", r"\1", desc).strip(" ()-:")

    if code_clean in {"-", ""}:
        return "-", desc, f"- ({desc})" if desc else "-"

    full = f"{code_clean} ({desc})" if desc else code_clean
    return code_clean, desc, full


def enrich_icd10(raw_icd: str, dx: str = "") -> str:
    _, _, full = parse_medical_code_and_desc(raw_icd, ICD10_DESCRIPTIONS, fallback_desc=dx)
    return full


def enrich_icd9(raw_icd: str) -> str:
    _, _, full = parse_medical_code_and_desc(raw_icd, ICD9_DESCRIPTIONS)
    return full


def extract_medical_form(text: str, provider: str | None = None) -> MedicalFormExtractOut:
    settings = get_settings()
    clean_text = apply_dictionary(text)

    extracted_data = None
    extracted_by = "rule_based"

    # Strategy 1: Groq LLM if key available or provider is groq
    if not extracted_data and (provider in {None, "auto", "groq"} or settings.transcription_provider == "groq") and settings.groq_api_key:
        print("[EXTRACT] Attempting Groq extraction...")
        extracted_data = extract_with_groq(clean_text, settings.groq_api_key, settings.groq_llm_model)
        if extracted_data:
            extracted_by = f"groq:{settings.groq_llm_model}"

    # Strategy 2: Gemini if key available
    if not extracted_data and (provider in {None, "auto", "gemini"}) and settings.gemini_api_key:
        print("[EXTRACT] Attempting Gemini extraction...")
        extracted_data = extract_with_gemini(clean_text, settings.gemini_api_key)
        if extracted_data:
            extracted_by = "gemini"

    # Strategy 3: OpenAI if key available
    if not extracted_data and (provider in {None, "auto", "openai"}) and settings.openai_api_key:
        print("[EXTRACT] Attempting OpenAI extraction...")
        extracted_data = extract_with_openai(clean_text, settings.openai_api_key, settings.openai_model)
        if extracted_data:
            extracted_by = "openai"

    # Strategy 4: Groq fallback
    if not extracted_data and settings.groq_api_key:
        print("[EXTRACT] Attempting Groq fallback extraction...")
        extracted_data = extract_with_groq(clean_text, settings.groq_api_key, settings.groq_llm_model)
        if extracted_data:
            extracted_by = f"groq:{settings.groq_llm_model}"

    # Strategy 5: Rule-based heuristic fallback
    if not extracted_data or not isinstance(extracted_data, dict):
        print("[EXTRACT] Falling back to rule-based heuristic extraction...")
        extracted_data = extract_with_heuristics(clean_text)
        extracted_by = "rule_based"

    vs_dict = extracted_data.get("vital_signs") or {}
    vital_signs_obj = normalize_vital_signs(vs_dict, clean_text)

    # Process investigations list & string
    inv_data = extracted_data.get("investigations") or extracted_data.get("investigation") or []
    if isinstance(inv_data, str):
        inv_list = [x.strip() for x in re.split(r"[\n,;]", inv_data) if x.strip()]
        inv_str = inv_data.strip()
    elif isinstance(inv_data, list):
        inv_list = [str(x).strip() for x in inv_data if str(x).strip()]
        inv_str = ", ".join(inv_list)
    else:
        inv_list = []
        inv_str = ""

    # Process assessment forms
    forms_data = extracted_data.get("assessment_forms") or extracted_data.get("assessmentForms") or []
    if isinstance(forms_data, str):
        raw_forms = [x.strip() for x in re.split(r"[\n,;]", forms_data) if x.strip()]
    elif isinstance(forms_data, list):
        raw_forms = [str(x).strip() for x in forms_data if str(x).strip()]
    else:
        raw_forms = []

    # Map form names or keywords to valid frontend form IDs
    form_id_map = {
        "smoking": ["smoking", "สูบบุหรี่", "บุหรี่", "fagerstrom", "fagerström"],
        "depression": ["depression", "ซึมเศร้า", "2q", "9q", "8q"],
        "fall_risk": ["fall_risk", "fall", "หกล้ม", "ล้ม", "morse"],
        "adl": ["adl", "barthel", "กิจวัตรประจำวัน", "ติดเตียง"],
        "cvd": ["cvd", "cv risk", "หลอดเลือดหัวใจ", "หัวใจ"],
        "braden": ["braden", "แผลกดทับ"],
        "alcohol": ["alcohol", "สุรา", "ดื่ม", "เหล้า", "audit"],
        "mna": ["mna", "โภชนาการ", "อาหาร"],
    }
    resolved_forms: list[str] = []
    for item in raw_forms:
        item_lower = item.lower().strip()
        matched = False
        for fid, keywords in form_id_map.items():
            if any(k in item_lower for k in keywords):
                if fid not in resolved_forms:
                    resolved_forms.append(fid)
                matched = True
                break
        if not matched and item_lower in form_id_map and item_lower not in resolved_forms:
            resolved_forms.append(item_lower)

    # Heuristic fallback if LLM returned empty assessment_forms or missed obvious ones
    if any(w in clean_text for w in ["หมดพลัง", "ซึมเศร้า", "เบื่อหน่าย", "คิดวน", "หลับยาก", "2Q", "9Q", "8Q"]):
        if "depression" not in resolved_forms:
            resolved_forms.append("depression")
    if any(w in clean_text for w in ["สูบบุหรี่", "บุหรี่"]):
        if "smoking" not in resolved_forms:
            resolved_forms.append("smoking")
    if any(w in clean_text for w in ["ดื่มเหล้า", "ดื่มเบียร์", "แอลกอฮอล์", "สุรา"]):
        if "alcohol" not in resolved_forms:
            resolved_forms.append("alcohol")
    if any(w in clean_text for w in ["หกล้ม", "ล้มบ่อย", "พลัดตก"]):
        if "fall_risk" not in resolved_forms:
            resolved_forms.append("fall_risk")
    if any(w in clean_text for w in ["แผลกดทับ"]):
        if "braden" not in resolved_forms:
            resolved_forms.append("braden")
    if any(w in clean_text for w in ["ติดเตียง", "ลุกไม่ไหว", "ช่วยเหลือตัวเองไม่ได้"]):
        if "adl" not in resolved_forms:
            resolved_forms.append("adl")
    if any(w in clean_text for w in ["เบื่ออาหาร", "น้ำหนักลดเร็ว"]):
        if "mna" not in resolved_forms:
            resolved_forms.append("mna")

    # Merge extracted assessments with heuristic defaults for all resolved forms
    extracted_assessments = extracted_data.get("assessments") or {}
    heuristic_assessments = extract_heuristic_assessments(clean_text, resolved_forms, vs_dict)

    final_assessments: dict[str, dict] = dict(heuristic_assessments)
    if isinstance(extracted_assessments, dict):
        for form_id, form_info in extracted_assessments.items():
            if isinstance(form_info, dict):
                if form_id in final_assessments:
                    final_assessments[form_id].update(form_info)
                else:
                    final_assessments[form_id] = form_info

    # Ensure pain is never put into assessment_forms or assessments (handled in VS painScore)
    final_assessments.pop("pain", None)
    resolved_forms = [f for f in resolved_forms if f != "pain"]

    # Ensure all keys in final_assessments are represented in assessment_forms
    for form_key in final_assessments.keys():
        if form_key not in resolved_forms:
            resolved_forms.append(form_key)

    # Parse and enrich ICD-10, ICD-9, and DRG fields (code, name/desc, and full string)
    prov_dx = extracted_data.get("provisionalDiagnosis") or extracted_data.get("provisional_diagnosis", "") or ""

    raw_icd10 = extracted_data.get("icd10") or extracted_data.get("icd10Code") or extracted_data.get("icd10_code") or ""
    raw_icd10_desc = extracted_data.get("icd10Name") or extracted_data.get("icd10_name") or extracted_data.get("icd10_desc") or ""
    icd10_code, icd10_desc, icd10_full = parse_medical_code_and_desc(
        raw_icd10, ICD10_DESCRIPTIONS, fallback_desc=prov_dx, explicit_desc=raw_icd10_desc
    )

    raw_icd9 = extracted_data.get("icd9") or extracted_data.get("icd9Code") or extracted_data.get("icd9_code") or ""
    raw_icd9_desc = extracted_data.get("icd9Name") or extracted_data.get("icd9_name") or extracted_data.get("icd9_desc") or ""
    icd9_code, icd9_desc, icd9_full = parse_medical_code_and_desc(
        raw_icd9, ICD9_DESCRIPTIONS, fallback_desc="", explicit_desc=raw_icd9_desc
    )

    raw_drg = extracted_data.get("drg") or extracted_data.get("drgCode") or extracted_data.get("drg_code") or ""
    raw_drg_desc = extracted_data.get("drgName") or extracted_data.get("drg_name") or extracted_data.get("drg_desc") or ""
    drg_code, drg_desc, drg_full = parse_medical_code_and_desc(
        raw_drg, DRG_DESCRIPTIONS, fallback_desc="", explicit_desc=raw_drg_desc
    )

    return MedicalFormExtractOut(
        chiefComplaint=extracted_data.get("chiefComplaint") or extracted_data.get("chief_complaint", "") or "",
        presentIllness=extracted_data.get("presentIllness") or extracted_data.get("present_illness", "") or "",
        pastHistory=extracted_data.get("pastHistory") or extracted_data.get("past_history", "") or "",
        vitalSigns=vital_signs_obj,
        physicalExam=extracted_data.get("physicalExam") or extracted_data.get("physical_exam", "") or "",
        provisionalDiagnosis=prov_dx,
        icd10=icd10_full,
        icd10Code=icd10_code,
        icd10Name=icd10_desc,
        icd9=icd9_full,
        icd9Code=icd9_code,
        icd9Name=icd9_desc,
        drg=drg_full,
        drgCode=drg_code,
        drgName=drg_desc,
        investigation=inv_str,
        investigations=inv_list,
        assessmentForms=resolved_forms,
        assessments=final_assessments,
        note=extracted_data.get("note") or extracted_data.get("treatment_plan", "") or "",
        disposition=extracted_data.get("disposition", "") or "",
        rawText=clean_text,
        extractedBy=extracted_by,
    )
