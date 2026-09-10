import os
import shutil
from pathlib import Path

# Disable Hugging Face symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import huggingface_hub.file_download

    _original_create_symlink = getattr(
        huggingface_hub.file_download,
        "_create_symlink",
        None,
    )

    if _original_create_symlink is not None:

        def _patched_create_symlink(src: str, dst: str, new_blob: bool = False):
            try:
                _original_create_symlink(src, dst, new_blob=new_blob)
            except OSError:
                # Windows non-admin [WinError 1314] fallback: copy file instead of symlink
                dst_path = Path(dst)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists() or dst_path.is_symlink():
                    try:
                        dst_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                if not os.path.isabs(src):
                    actual_src = (dst_path.parent / src).resolve()
                else:
                    actual_src = Path(src)

                if actual_src.exists():
                    shutil.copy2(str(actual_src), str(dst_path))

        huggingface_hub.file_download._create_symlink = _patched_create_symlink
        print("HF symlink patch applied for Windows compatibility")

except Exception as err:
    print(f"Could not patch huggingface_hub symlink: {err}")
