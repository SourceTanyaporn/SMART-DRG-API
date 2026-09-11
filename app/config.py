import os
from functools import lru_cache
from multiprocessing import cpu_count

from dotenv import load_dotenv

load_dotenv(override=True)


class Settings:
    app_name: str = os.getenv("APP_NAME", "Whisper API")
    model_size: str = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
    device: str = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    upload_dir: str = os.getenv("UPLOAD_DIR", "/tmp/whisper-uploads")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "200"))
    cpu_threads: int = int(os.getenv("WHISPER_CPU_THREADS", str(max(2, cpu_count() // 2))))
    num_workers: int = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
    default_beam_size: int = int(os.getenv("WHISPER_BEAM_SIZE", "2"))
    default_vad_filter: bool = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
    condition_on_previous_text: bool = os.getenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
    repetition_penalty: float = float(os.getenv("WHISPER_REPETITION_PENALTY", "1.0"))
    no_repeat_ngram_size: int = int(os.getenv("WHISPER_NO_REPEAT_NGRAM_SIZE", "0"))
    initial_prompt: str | None = os.getenv(
        "WHISPER_INITIAL_PROMPT",
        "บทสนทนาการตรวจรักษาทางการแพทย์ คุณหมอ แพทย์ คนไข้ ผู้ป่วย หมอเห็นผล ซักประวัติ ตรวจร่างกาย สัญญาณชีพ ความดัน ชีพจร อุณหภูมิ อาการ วินิจฉัย การรักษา ยา และคำแนะนำทางการแพทย์",
    ) or None

    post_process_mode: str = os.getenv("TRANSCRIPT_POST_PROCESS", "dictionary")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    num_speakers: int | None = int(os.getenv("WHISPER_NUM_SPEAKERS")) if os.getenv("WHISPER_NUM_SPEAKERS") else None
    min_speakers: int | None = int(os.getenv("WHISPER_MIN_SPEAKERS")) if os.getenv("WHISPER_MIN_SPEAKERS") else None
    max_speakers: int | None = int(os.getenv("WHISPER_MAX_SPEAKERS")) if os.getenv("WHISPER_MAX_SPEAKERS") else None
    whisperx_batch_size: int = int(os.getenv("WHISPERX_BATCH_SIZE", "4"))
    huggingface_token: str | None = os.getenv("HUGGINGFACE_TOKEN") or None
    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    groq_whisper_model: str = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
    groq_llm_model: str = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    transcription_provider: str = os.getenv("TRANSCRIPTION_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "local")


def get_settings() -> Settings:
    load_dotenv(override=True)
    return Settings()

