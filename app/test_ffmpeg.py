from app.ffmpeg_config import FFMPEG_BIN

import shutil

print("FFmpeg folder:")
print(FFMPEG_BIN)

print()

print("FFmpeg executable:")
print(shutil.which("ffmpeg"))