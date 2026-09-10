import subprocess
from pathlib import Path

from app.ffmpeg_config import FFMPEG_BIN


def convert_to_wav(input_path: Path) -> Path:
    output_path = input_path.with_suffix(".wav")

    command = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    print("========== FFMPEG CONVERSION ==========")
    print("INPUT:", input_path)
    print("OUTPUT:", output_path)
    print("FFMPEG:", FFMPEG_BIN)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("FFMPEG ERROR:")
        print(result.stderr)

        raise RuntimeError(
            f"FFmpeg conversion failed: {result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"FFmpeg did not create WAV file: {output_path}"
        )

    print("WAV CREATED:", output_path)
    print("WAV SIZE:", output_path.stat().st_size)

    return output_path