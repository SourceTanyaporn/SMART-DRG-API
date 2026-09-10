import app.ffmpeg_config
import app.hf_patch
from pathlib import Path
import subprocess
import shutil

import torch
import soundfile as sf

from pyannote.audio import Pipeline

from app.config import get_settings


# ============================================================
# GLOBAL
# ============================================================

_pipeline = None


# ============================================================
# PYANNOTE PIPELINE
# ============================================================

def get_diarization_pipeline():
    """
    Load Pyannote speaker diarization pipeline.

    Pipeline จะถูกโหลดเพียงครั้งเดียว
    แล้วนำกลับมาใช้ซ้ำใน job ถัดไป
    """

    global _pipeline

    settings = get_settings()

    print("========== GET DIARIZATION PIPELINE ==========")
    print("DEVICE:", settings.device)
    print(
        "HF TOKEN EXISTS:",
        bool(settings.huggingface_token)
    )

    if not settings.huggingface_token:
        raise RuntimeError(
            "ไม่พบ HUGGINGFACE_TOKEN ใน environment"
        )

    # --------------------------------------------------------
    # Load pipeline only once
    # --------------------------------------------------------

    if _pipeline is None:

        print("LOADING PYANNOTE PIPELINE...")

        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=settings.huggingface_token,
        )

        if _pipeline is None:
            raise RuntimeError(
                "ไม่สามารถโหลด Pyannote Pipeline ได้"
            )

        # ----------------------------------------------------
        # Set device
        # ----------------------------------------------------

        device = torch.device(settings.device)

        print("PYANNOTE DEVICE:", device)

        _pipeline.to(device)

        print("PYANNOTE PIPELINE LOADED")

    else:

        print(
            "PYANNOTE PIPELINE ALREADY LOADED"
        )

    return _pipeline


# ============================================================
# FFMPEG
# ============================================================

def get_ffmpeg_path() -> Path:
    """
    Return FFmpeg executable path.
    """
    from app.ffmpeg_config import FFMPEG_EXE

    if FFMPEG_EXE.exists():
        return FFMPEG_EXE

    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return Path(which_ffmpeg)

    raise FileNotFoundError(
        f"FFmpeg not found: {FFMPEG_EXE}"
    )


# ============================================================
# CONVERT AUDIO -> WAV
# ============================================================

def convert_to_wav(audio_path: Path) -> Path:
    """
    Convert audio file to:

    - WAV
    - Mono
    - 16kHz
    - PCM 16-bit
    """

    audio_path = Path(audio_path)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if audio_path.stat().st_size == 0:
        raise RuntimeError(
            f"Audio file is empty: {audio_path}"
        )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    wav_path = audio_path.with_suffix(".wav")

    # --------------------------------------------------------
    # FFmpeg
    # --------------------------------------------------------

    ffmpeg = get_ffmpeg_path()

    print("========== FFMPEG CONVERSION ==========")
    print("INPUT:", audio_path)
    print("OUTPUT:", wav_path)
    print("FFMPEG:", ffmpeg)

    # --------------------------------------------------------
    # Run FFmpeg
    # --------------------------------------------------------

    result = subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # --------------------------------------------------------
    # FFmpeg error
    # --------------------------------------------------------

    if result.returncode != 0:

        print("========== FFMPEG ERROR ==========")
        print(result.stderr)

        raise RuntimeError(
            "FFmpeg conversion failed "
            f"with code {result.returncode}"
        )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if not wav_path.exists():
        raise RuntimeError(
            f"WAV conversion failed: {wav_path}"
        )

    if wav_path.stat().st_size == 0:
        raise RuntimeError(
            f"WAV file is empty: {wav_path}"
        )

    print("WAV CREATED:", wav_path)
    print(
        "WAV SIZE:",
        wav_path.stat().st_size
    )

    return wav_path


# ============================================================
# LOAD WAV
# ============================================================

def load_wav_as_tensor(wav_path: Path):
    """
    Load WAV using soundfile instead of torchaudio/TorchCodec.

    Returns:

        waveform:
            torch.Tensor
            shape = [channels, samples]

        sample_rate:
            int
    """

    print("========== LOAD WAV ==========")
    print("WAV:", wav_path)
    print("WAV EXISTS:", wav_path.exists())

    if not wav_path.exists():
        raise FileNotFoundError(
            f"WAV file not found: {wav_path}"
        )

    # --------------------------------------------------------
    # Read WAV
    # --------------------------------------------------------

    audio, sample_rate = sf.read(
        str(wav_path),
        dtype="float32",
    )

    print("SAMPLE RATE:", sample_rate)
    print("AUDIO SHAPE:", audio.shape)
    print("AUDIO DTYPE:", audio.dtype)

    # --------------------------------------------------------
    # Validate sample rate
    # --------------------------------------------------------

    if sample_rate != 16000:
        raise RuntimeError(
            f"Expected 16000Hz but got {sample_rate}Hz"
        )

    # --------------------------------------------------------
    # Convert NumPy -> Torch Tensor
    # --------------------------------------------------------

    waveform = torch.from_numpy(audio)

    # --------------------------------------------------------
    # Convert shape
    #
    # soundfile:
    #
    # Mono:
    #   [samples]
    #
    # Stereo:
    #   [samples, channels]
    #
    # Pyannote:
    #   [channels, samples]
    # --------------------------------------------------------

    if waveform.ndim == 1:

        # Mono
        waveform = waveform.unsqueeze(0)

    elif waveform.ndim == 2:

        # [samples, channels]
        # ->
        # [channels, samples]

        waveform = waveform.transpose(0, 1)

    else:

        raise RuntimeError(
            f"Unexpected audio shape: "
            f"{waveform.shape}"
        )

    # --------------------------------------------------------
    # Make sure audio is mono
    # --------------------------------------------------------

    if waveform.shape[0] > 1:

        print(
            "MULTI CHANNEL AUDIO DETECTED:",
            waveform.shape[0]
        )

        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    # --------------------------------------------------------
    # Validate waveform
    # --------------------------------------------------------

    if waveform.numel() == 0:
        raise RuntimeError(
            "WAV contains no audio samples"
        )

    print(
        "WAVEFORM SHAPE:",
        waveform.shape
    )

    print(
        "WAVEFORM DTYPE:",
        waveform.dtype
    )

    return waveform, sample_rate


# ============================================================
# EXTRACT DIARIZATION SEGMENTS
# ============================================================

def extract_speaker_segments(diarization):
    """
    Extract speaker segments from Pyannote output.

    รองรับทั้ง:

    1. Annotation
       - Pyannote รุ่นเก่า

    2. DiarizeOutput
       - Pyannote รุ่นใหม่
    """

    print(
        "========== STEP 4: EXTRACT SPEAKERS =========="
    )

    print(
        "DIARIZATION TYPE:",
        type(diarization)
    )

    # --------------------------------------------------------
    # Pyannote รุ่นใหม่
    #
    # DiarizeOutput
    #
    # ต้องดึง speaker_diarization ออกมาก่อน
    # --------------------------------------------------------

    if hasattr(
        diarization,
        "speaker_diarization"
    ):

        print(
            "PYANNOTE OUTPUT: DiarizeOutput"
        )

        speaker_diarization = (
            diarization.speaker_diarization
        )

    # --------------------------------------------------------
    # Pyannote รุ่นเก่า
    #
    # Annotation
    # --------------------------------------------------------

    else:

        print(
            "PYANNOTE OUTPUT: Annotation"
        )

        speaker_diarization = diarization

    # --------------------------------------------------------
    # Validate itertracks
    # --------------------------------------------------------

    if not hasattr(
        speaker_diarization,
        "itertracks"
    ):

        raise RuntimeError(
            "ไม่สามารถอ่าน speaker diarization output ได้ "
            f"type={type(speaker_diarization)}"
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    segments = []

    for (
        turn,
        _,
        speaker
    ) in speaker_diarization.itertracks(
        yield_label=True
    ):

        segment = {
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        }

        segments.append(segment)

        print(
            f"{turn.start:.2f} - "
            f"{turn.end:.2f} : "
            f"{speaker}"
        )

    print(
        "SPEAKER SEGMENTS:",
        len(segments)
    )

    return segments


# ============================================================
# DIARIZATION
# ============================================================

def diarize_audio(
    audio_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    """
    Run speaker diarization.

    Flow:

        MP3
          ↓
        FFmpeg
          ↓
        WAV 16kHz Mono
          ↓
        soundfile
          ↓
        Torch Tensor
          ↓
        Pyannote
          ↓
        DiarizeOutput / Annotation
          ↓
        Speaker segments
          ↓
        Delete temporary WAV
    """

    audio_path = Path(audio_path)
    settings = get_settings()

    print("========== DIARIZE AUDIO ==========")
    print("AUDIO:", audio_path)
    print(
        "AUDIO EXISTS:",
        audio_path.exists()
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if audio_path.stat().st_size == 0:
        raise RuntimeError(
            f"Audio file is empty: {audio_path}"
        )

    # --------------------------------------------------------
    # Get Pyannote pipeline
    # --------------------------------------------------------

    pipeline = get_diarization_pipeline()

    wav_path = None

    try:

        # ====================================================
        # STEP 1
        # MP3 -> WAV
        # ====================================================

        print(
            "========== STEP 1: CONVERT AUDIO =========="
        )

        wav_path = convert_to_wav(
            audio_path
        )

        print(
            "STEP: WAV READY"
        )

        print(
            "WAV EXISTS:",
            wav_path.exists()
        )

        # ====================================================
        # STEP 2
        # Load WAV
        # ====================================================

        print(
            "========== STEP 2: LOAD WAV =========="
        )

        waveform, sample_rate = (
            load_wav_as_tensor(
                wav_path
            )
        )

        # ====================================================
        # STEP 3
        # Run Pyannote
        # ====================================================

        print(
            "========== STEP 3: RUN PYANNOTE =========="
        )

        print(
            "WAVEFORM SHAPE:",
            waveform.shape
        )

        print(
            "SAMPLE RATE:",
            sample_rate
        )

        resolved_num_speakers = (
            num_speakers if num_speakers is not None else settings.num_speakers
        )
        resolved_min_speakers = (
            min_speakers if min_speakers is not None else settings.min_speakers
        )
        resolved_max_speakers = (
            max_speakers if max_speakers is not None else settings.max_speakers
        )

        params = {}
        if resolved_num_speakers is not None and resolved_num_speakers > 0:
            params["num_speakers"] = resolved_num_speakers
        else:
            if resolved_min_speakers is not None and resolved_min_speakers > 0:
                params["min_speakers"] = resolved_min_speakers
            if resolved_max_speakers is not None and resolved_max_speakers > 0:
                params["max_speakers"] = resolved_max_speakers

        print("PYANNOTE PARAMS:", params)

        diarization = pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            **params,
        )

        print(
            "STEP: PIPELINE COMPLETED"
        )

        print(
            "PYANNOTE RESULT TYPE:",
            type(diarization)
        )

        # ====================================================
        # STEP 4
        # Extract speakers
        # ====================================================

        segments = extract_speaker_segments(
            diarization
        )

        # ====================================================
        # STEP 5
        # Return
        # ====================================================

        print(
            "========== DIARIZATION COMPLETED =========="
        )

        print(
            "TOTAL SEGMENTS:",
            len(segments)
        )

        return segments

    finally:

        # ====================================================
        # Cleanup temporary WAV
        # ====================================================

        if (
            wav_path is not None
            and wav_path.exists()
        ):

            try:

                wav_path.unlink()

                print(
                    "TEMP WAV DELETED:",
                    wav_path
                )

            except Exception as cleanup_error:

                print(
                    "WARNING: "
                    "Could not delete WAV:",
                    cleanup_error
                )

