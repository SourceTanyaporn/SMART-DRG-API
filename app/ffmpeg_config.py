import os
from pathlib import Path
import app.hf_patch

BASE_DIR = Path(__file__).resolve().parent.parent
FFMPEG_DIR = BASE_DIR / "ffmpeg" / "bin"
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"

if FFMPEG_DIR.exists():
    ffmpeg_dir_str = str(FFMPEG_DIR)
    if ffmpeg_dir_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir_str + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(ffmpeg_dir_str)
        except Exception:
            pass

FFMPEG_BIN = str(FFMPEG_EXE) if FFMPEG_EXE.exists() else "ffmpeg"