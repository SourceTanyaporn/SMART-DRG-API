import os
import sys
from dotenv import load_dotenv
load_dotenv()

from app.config import get_settings
from app.postprocess import call_groq_proofread_json, post_process_segments
from app.schemas import SegmentOut

settings = get_settings()
print("GROQ KEY exists:", bool(settings.groq_api_key))
print("GROQ MODEL:", settings.groq_llm_model)

sample_segments = [
    SegmentOut(id=0, start=0.0, end=5.0, text="สวัสดีค่ะ เชิญนั่งก่อนค่ะ ว่าเป็นยังไงบ้างค่ะช่วงนี้ สบายดีไหม", speaker="SPEAKER_00", role=None),
    SegmentOut(id=1, start=5.1, end=10.0, text="สวัสดีค่ะคุณหมอก็เรื่อยๆค่ะแต่รู้ซึกเพียร่าๆ ตลอดเวลาเลยค่ะ", speaker="SPEAKER_00", role=None),
    SegmentOut(id=2, start=10.1, end=20.0, text="มอเห็นผลที่พยาบาล วัดความดันค่ะกับชีพจรเต้น ความดันตัวบนตรวจ S-142 ชีพจรเต้นว่าเป็นค่อนข้างเร็ว", speaker="SPEAKER_00", role=None),
]

payload = [
    {"id": s.id, "speaker": s.speaker, "role": s.role or "หมอ", "text": s.text}
    for s in sample_segments
]

res = call_groq_proofread_json(payload, settings.groq_api_key, settings.groq_llm_model)
print("call_groq_proofread_json result:", res)

final_segs, full_text, raw_text, by = post_process_segments(sample_segments, mode="groq")
print("post_processed_by:", by)
print("final segments count:", len(final_segs))
for s in final_segs:
    print(f"[{s.role or s.speaker}] ({s.start}-{s.end}): {s.text}")
