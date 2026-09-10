from pydantic import BaseModel, Field


class SegmentOut(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    role: str | None = None


class TranscriptionOut(BaseModel):
    text: str
    raw_text: str | None = None
    post_processed_by: str | None = None
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    segments: list[SegmentOut] = Field(default_factory=list)


class PostProcessIn(BaseModel):
    text: str
    post_process: str | None = None


class PostProcessOut(BaseModel):
    text: str
    raw_text: str
    post_processed_by: str

class TranscriptionJobOut(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    result: TranscriptionOut | None = None
    error: str | None = None


class VitalSignsData(BaseModel):
    # 1. Pulse (from pr = pulse)
    pulse: str | None = None

    # 2. Body Temperature (from temperature = bodyTemperature)
    bodyTemperature: str | None = None

    # 3. BP (from blood_pressure = bp, bp1_systolic = systolic, bp1_diastolic = diastolic)
    bp: str | None = None
    systolic: str | None = None
    diastolic: str | None = None

    # 4. BP2 (from blood_pressure_2 = bp2, bp2_systolic = systolic2, bp2_diastolic = diastolic2)
    bp2: str | None = None
    systolic2: str | None = None
    diastolic2: str | None = None

    # 5. O2sat (from spo2 = o2sat)
    o2sat: str | None = None

    # 6. MAP & MAP2
    map: str | None = None
    map2: str | None = None

    # 7. Weight & Height & BMI
    weight: str | None = None
    height: str | None = None
    bmi: str | None = None

    # 8. Body measurements (chest, waist)
    chest: str | None = None
    waist: str | None = None

    # 9. Additional metrics (respiratory, painScore)
    respiratory: str | None = None
    painScore: str | None = None


class MedicalFormExtractIn(BaseModel):
    text: str
    provider: str | None = None


class MedicalFormExtractOut(BaseModel):
    chiefComplaint: str = ""
    presentIllness: str = ""
    pastHistory: str = ""
    vitalSigns: VitalSignsData = Field(default_factory=VitalSignsData)
    physicalExam: str = ""
    provisionalDiagnosis: str = ""
    icd10: str = ""
    icd10Code: str = ""
    icd10Name: str = ""
    icd9: str = ""
    icd9Code: str = ""
    icd9Name: str = ""
    drg: str = ""
    drgCode: str = ""
    drgName: str = ""
    investigation: str = ""
    investigations: list[str] = Field(default_factory=list)
    assessmentForms: list[str] = Field(default_factory=list)
    assessments: dict[str, dict] = Field(default_factory=dict)
    note: str = ""
    disposition: str = ""
    rawText: str = ""
    extractedBy: str = "rule_based"


class ClinicalChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class PatientContext(BaseModel):
    patient_name: str | None = None
    hn: str | None = None
    age: str | None = None
    gender: str | None = None
    raw_transcript: str | None = None
    vital_signs: dict[str, str | None] | None = None
    current_medications: list[str] | None = None
    current_diagnosis: str | None = None
    allergies: str | None = None
    underlying_diseases: str | None = None


class ClinicalChatIn(BaseModel):
    messages: list[ClinicalChatMessage] = Field(default_factory=list)
    patient_context: PatientContext | None = None
    provider: str | None = None  # "gemini" or "openai"
    mode: str | None = "general"  # "general", "soap", "drug_safety", "icd_drg", "diff_dx"


class ClinicalChatOut(BaseModel):
    reply: str
    structured_action: dict | None = None
    suggested_quick_prompts: list[str] = Field(default_factory=list)
    provider_used: str = "gemini"