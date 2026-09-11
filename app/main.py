import app.ffmpeg_config
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks

from app.config import get_settings
from app.postprocess import post_process_transcript
from app.schemas import (
    PostProcessIn,
    PostProcessOut,
    TranscriptionOut,
    MedicalFormExtractIn,
    MedicalFormExtractOut,
    ClinicalChatIn,
    ClinicalChatOut,
)
from app.extraction_service import extract_medical_form
from app.chat_service import run_clinical_chat
from app.whisper_service import find_speaker, get_model, transcribe_audio, transcribe_by_speaker
from app.diarization_service import diarize_audio
from app.whisperx_service import run_whisperx_pipeline
from app.alignment_service import (
    align_transcript_with_speakers,
    merge_same_speaker_segments,
)
from app.job_manager import (
    create_job,
    get_job,
    update_job,
)

settings = get_settings()

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "online",
        "message": "SMART-DRG Whisper & Clinical AI API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "provider": settings.transcription_provider,
        "model": settings.groq_whisper_model if settings.transcription_provider == "groq" else settings.model_size,
        "device": settings.device,
    }


@app.post("/v1/warmup")
def warmup() -> dict[str, str]:
    if settings.groq_api_key or settings.transcription_provider in {"groq", "openai"}:
        return {"status": "ready", "provider": settings.transcription_provider}
    get_model()
    return {"status": "ready", "model": settings.model_size}


@app.post("/v1/postprocess", response_model=PostProcessOut)
def postprocess(payload: PostProcessIn) -> PostProcessOut:
    target_mode = payload.post_process
    if target_mode == "openai" and not settings.openai_api_key:
        target_mode = settings.post_process_mode or "groq"

    corrected_text, post_processed_by = post_process_transcript(
        payload.text,
        target_mode,
    )
    return PostProcessOut(
        text=corrected_text,
        raw_text=payload.text,
        post_processed_by=post_processed_by,
    )


@app.post("/v1/extract-form", response_model=MedicalFormExtractOut)
def extract_form(payload: MedicalFormExtractIn) -> MedicalFormExtractOut:
    return extract_medical_form(
        payload.text,
        provider=payload.provider,
        allergies=payload.allergies,
        patient_name=payload.patient_name,
    )


@app.post("/v1/chat/clinical", response_model=ClinicalChatOut)
def clinical_chat(payload: ClinicalChatIn) -> ClinicalChatOut:
    return run_clinical_chat(payload)


# @app.post("/v1/transcribe", response_model=TranscriptionOut)
# async def transcribe(
#     file: UploadFile = File(...),
#     language: str | None = Form(default=None),
#     task: str = Form(default="transcribe"),
#     beam_size: int | None = Form(default=None),
#     vad_filter: bool | None = Form(default=None),
#     initial_prompt: str | None = Form(default=None),
#     post_process: str | None = Form(default=None),
# ) -> TranscriptionOut:
#     resolved_beam_size = beam_size or settings.default_beam_size
#     resolved_vad_filter = (
#         vad_filter if vad_filter is not None else settings.default_vad_filter
#     )

#     if post_process == "openai" and not settings.openai_api_key:
#         raise HTTPException(
#             status_code=400,
#             detail="OPENAI_API_KEY is required when post_process=openai",
#         )

#     if task not in {"transcribe", "translate"}:
#         raise HTTPException(status_code=400, detail="task must be transcribe or translate")

#     if resolved_beam_size < 1 or resolved_beam_size > 10:
#         raise HTTPException(status_code=400, detail="beam_size must be between 1 and 10")

#     content = await file.read()
#     max_bytes = settings.max_upload_mb * 1024 * 1024
#     if len(content) > max_bytes:
#         raise HTTPException(
#             status_code=413,
#             detail=f"file is larger than {settings.max_upload_mb} MB",
#         )

#     suffix = Path(file.filename or "audio").suffix or ".audio"
#     temp_path: Path | None = None
#     try:
#         with NamedTemporaryFile(
#             suffix=suffix,
#             dir=settings.upload_dir,
#             delete=False,
#         ) as temp_file:
#             temp_file.write(content)
#             temp_path = Path(temp_file.name)

#         return transcribe_audio(
#             temp_path,
#             language=language,
#             task=task,
#             beam_size=resolved_beam_size,
#             vad_filter=resolved_vad_filter,
#             initial_prompt=initial_prompt or settings.initial_prompt,
#             post_process=post_process,
#         )
#     finally:
#         if temp_path and temp_path.exists():
#             temp_path.unlink(missing_ok=True)
@app.post("/v1/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str | None = Form(default="th"),
    task: str = Form(default="transcribe"),
    beam_size: int | None = Form(default=None),
    vad_filter: bool | None = Form(default=None),
    initial_prompt: str | None = Form(default=None),
    post_process: str | None = Form(default=None),
):
    resolved_beam_size = beam_size or settings.default_beam_size

    resolved_vad_filter = (
        vad_filter
        if vad_filter is not None
        else settings.default_vad_filter
    )

    resolved_post_process = post_process
    if resolved_post_process == "openai" and not settings.openai_api_key:
        resolved_post_process = settings.post_process_mode or "groq"

    if task not in {"transcribe", "translate"}:
        raise HTTPException(
            status_code=400,
            detail="task must be transcribe or translate",
        )

    if resolved_beam_size < 1 or resolved_beam_size > 10:
        raise HTTPException(
            status_code=400,
            detail="beam_size must be between 1 and 10",
        )

    content = await file.read()

    max_bytes = settings.max_upload_mb * 1024 * 1024

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file is larger than {settings.max_upload_mb} MB",
        )

    suffix = Path(file.filename or "audio").suffix or ".audio"

    temp_path: Path | None = None

    try:

        with NamedTemporaryFile(
            suffix=suffix,
            dir=settings.upload_dir,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        print("========== FILE CREATED ==========")
        print("PATH:", temp_path)
        print("EXISTS:", temp_path.exists())
        print("SIZE:", temp_path.stat().st_size if temp_path.exists() else 0)
        job_id = create_job()

        print("========== JOB CREATED ==========")
        print("JOB ID:", job_id)
        print("TEMP FILE:", temp_path)
        print("FILE EXISTS:", temp_path.exists())

        background_tasks.add_task(
            process_transcription,
            job_id,
            temp_path,
            language,
            task,
            resolved_beam_size,
            resolved_vad_filter,
            initial_prompt or settings.initial_prompt,
            resolved_post_process,
        )
        return {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "รอประมวลผล",
        }

    except Exception:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

        raise

@app.get("/v1/transcribe/status/{job_id}")
def transcription_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return job
def process_transcription(
    job_id: str,
    temp_path: Path,
    language: str | None,
    task: str,
    beam_size: int,
    vad_filter: bool,
    initial_prompt: str | None,
    post_process: str | None,
):
    try:
        print("========== START JOB (WHISPERX PIPELINE) ==========")
        print("JOB ID:", job_id)
        print("FILE:", temp_path)

        def on_progress(progress: int, message: str):
            print(f"PROGRESS [{progress}%]: {message}")
            update_job(
                job_id,
                status="processing",
                progress=progress,
                message=message,
            )

        result = run_whisperx_pipeline(
            temp_path,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
            post_process=post_process,
            progress_callback=on_progress,
        )

        print("========== WHISPERX PIPELINE COMPLETED ==========")
        print(f"Total segments: {len(result.get('segments', []))}")

        update_job(
            job_id,
            status="completed",
            progress=100,
            message="แปลงเสียงและแยกผู้พูดเสร็จแล้ว",
            result=result,
        )

    except Exception as exc:
        print("========== JOB ERROR ==========")
        print(type(exc).__name__)
        print(exc)
        traceback.print_exc()

        update_job(
            job_id,
            status="failed",
            progress=0,
            message="เกิดข้อผิดพลาด",
            error=str(exc),
        )

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        print("========== END JOB ==========")