from functools import lru_cache
from pathlib import Path
from collections.abc import Callable
import subprocess
from tempfile import NamedTemporaryFile

from faster_whisper import WhisperModel

from app.config import get_settings
from app.postprocess import post_process_transcript
from app.schemas import SegmentOut, TranscriptionOut
from app.diarization_service import diarize_audio


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    settings = get_settings()

    return WhisperModel(
        settings.model_size,
        device=settings.device,
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
        num_workers=settings.num_workers,
    )

def find_speaker(
    start: float,
    end: float,
    speakers: list[dict],
) -> str | None:

    if not speakers:
        return None

    best_speaker = None
    best_overlap = 0.0

    for item in speakers:

        speaker_start = float(
            item["start"]
        )

        speaker_end = float(
            item["end"]
        )

        overlap_start = max(
            start,
            speaker_start,
        )

        overlap_end = min(
            end,
            speaker_end,
        )

        overlap = max(
            0.0,
            overlap_end - overlap_start,
        )

        if overlap > best_overlap:

            best_overlap = overlap
            best_speaker = item["speaker"]

    return best_speaker
def cut_audio_segment(
    audio_path: Path,
    start: float,
    end: float,
) -> Path:

    output_file = NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )

    output_path = Path(output_file.name)

    output_file.close()

    ffmpeg_path = r"C:\Users\syste\Documents\whisper\ffmpeg\bin\ffmpeg.exe"

    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return output_path

# def transcribe_audio(
#     audio_path: Path,
#     *,
#     language: str | None,
#     task: str,
#     beam_size: int,
#     vad_filter: bool,
#     initial_prompt: str | None,
#     post_process: str | None,
#     progress_callback: Callable[[int], None] | None = None,
# ) -> TranscriptionOut:

#     print("========== WHISPER TRANSCRIPTION START ==========")
#     print("Audio:", audio_path)

#     segments_iter, info = get_model().transcribe(
#         str(audio_path),
#         language=language,
#         task=task,
#         beam_size=beam_size,
#         vad_filter=vad_filter,
#         initial_prompt=initial_prompt,
#     )

#     segments = []
#     texts = []

#     duration = getattr(info, "duration", None)

#     for index, segment in enumerate(segments_iter):

#         text = segment.text.strip()

#         if not text:
#             continue

#         start = float(segment.start)
#         end = float(segment.end)

#         print(
#             f"WHISPER: "
#             f"{start:.3f} - {end:.3f} | "
#             f"{text}"
#         )

#         segments.append(
#             SegmentOut(
#                 id=index,
#                 start=round(start, 3),
#                 end=round(end, 3),
#                 text=text,
#                 speaker=None,
#                 role=None,
#             )
#         )

#         texts.append(text)

#         if progress_callback and duration:
#             ratio = min(end / duration, 1.0)

#             progress = 30 + int(ratio * 55)

#             progress_callback(progress)

#     raw_text = " ".join(texts).strip()

#     corrected_text, post_processed_by = (
#         post_process_transcript(
#             raw_text,
#             post_process,
#         )
#     )

#     return TranscriptionOut(
#         text=corrected_text,
#         raw_text=raw_text,
#         post_processed_by=post_processed_by or "none",
#         language=getattr(info, "language", None),
#         language_probability=getattr(
#             info,
#             "language_probability",
#             None,
#         ),
#         duration=duration,
#         segments=segments,
#     )
def transcribe_audio(
    audio_path: Path,
    *,
    language: str | None,
    task: str,
    beam_size: int,
    vad_filter: bool,
    initial_prompt: str | None,
    post_process: str | None,
    progress_callback: Callable[[int], None] | None = None,
) -> TranscriptionOut:

    print("========== WHISPER TRANSCRIPTION START ==========")
    print("Audio:", audio_path)

    if progress_callback:
        progress_callback(15)

    # ========================================
    # WHISPER
    # ========================================

    settings = get_settings()
    segments_iter, info = get_model().transcribe(
        str(audio_path),
        language=language,
        task=task,
        beam_size=beam_size or settings.default_beam_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt or settings.initial_prompt,
        condition_on_previous_text=settings.condition_on_previous_text,
        repetition_penalty=settings.repetition_penalty,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
        temperature=[0.0, 0.2],
        word_timestamps=True,
    )


    print("WHISPER TRANSCRIBE CREATED")

    print(
        "Language:",
        getattr(info, "language", None),
    )

    print(
        "Language probability:",
        getattr(
            info,
            "language_probability",
            None,
        ),
    )

    print(
        "Duration:",
        getattr(info, "duration", None),
    )

    segments = []
    texts = []

    duration = getattr(
        info,
        "duration",
        None,
    )

    last_progress = 30

    # ========================================
    # WHISPER SEGMENTS
    # ========================================

    for index, segment in enumerate(
        segments_iter
    ):

        text = segment.text.strip()

        if not text:
            continue

        start = float(segment.start)
        end = float(segment.end)

        print(
            f"WHISPER SEGMENT {index}: "
            f"{start:.3f} - "
            f"{end:.3f} | "
            f"{text}"
        )

        texts.append(text)

        segments.append(
            SegmentOut(
                id=index,
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                speaker=None,
                role=None,
            )
        )

        # ========================================
        # PROGRESS
        # ========================================

        if progress_callback and duration:

            ratio = min(
                end / duration,
                1.0,
            )

            progress = 30 + int(
                ratio * 55
            )

            if progress > last_progress:

                last_progress = progress

                progress_callback(
                    progress
                )

    # ========================================
    # RAW TEXT
    # ========================================

    raw_text = " ".join(
        texts
    ).strip()

    if progress_callback:
        progress_callback(90)

    print(
        "========== WHISPER TRANSCRIPTION COMPLETED =========="
    )

    # ========================================
    # POST PROCESS
    # ========================================
    #
    # ตอนนี้ไม่ให้ OpenAI แก้
    # เพื่อทดสอบ Whisper ก่อน
    #

    corrected_text = raw_text
    post_processed_by = ""

    if post_process:
        print(
            "POST PROCESS REQUESTED:",
            post_process,
        )

        corrected_text, post_processed_by = (
            post_process_transcript(
                raw_text,
                post_process,
            )
        )

    if progress_callback:
        progress_callback(98)

    return TranscriptionOut(
        text=corrected_text,
        raw_text=raw_text,

        # ต้องเป็น string ไม่ใช่ None
        post_processed_by=post_processed_by,

        language=getattr(
            info,
            "language",
            None,
        ),

        language_probability=getattr(
            info,
            "language_probability",
            None,
        ),

        duration=duration,

        segments=segments,
    )

def merge_diarization_segments(
    segments: list,
    max_gap: float = 0.3,
):
    if not segments:
        return []

    # sort ตามเวลา
    segments = sorted(
        segments,
        key=lambda x: float(x["start"])
    )

    merged = []

    for segment in segments:

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        speaker = segment["speaker"]

        if end <= start:
            continue

        if not merged:

            merged.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                }
            )

            continue

        current = merged[-1]

        same_speaker = (
            current["speaker"]
            == speaker
        )

        gap = start - current["end"]

        if same_speaker and gap <= max_gap:

            current["end"] = max(
                current["end"],
                end,
            )

        else:

            merged.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                }
            )

    return merged

def transcribe_by_speaker(
    temp_path: Path,
    speaker_segments: list,
    language: str | None,
    task: str,
    beam_size: int,
    vad_filter: bool,
    initial_prompt: str | None,
    post_process: str | None,
    progress_callback=None,
):
    results = []

    if not speaker_segments:
        print("NO SPEAKER SEGMENTS")
        return results

    total = len(speaker_segments)

    print("========== TRANSCRIBE BY SPEAKER ==========")
    print("TOTAL SPEAKER SEGMENTS:", total)

    for index, speaker_segment in enumerate(speaker_segments):

        start = float(speaker_segment["start"])
        end = float(speaker_segment["end"])
        speaker = speaker_segment["speaker"]

        print(
            f"========== SPEAKER {index + 1}/{total} =========="
        )
        print(f"SPEAKER: {speaker}")
        print(f"TIME: {start:.3f} - {end:.3f}")

        if end <= start:
            continue

        segment_audio = None

        try:
            # ตัดเสียงตามช่วงของ speaker
            segment_audio = cut_audio_segment(
                temp_path,
                start,
                end,
            )

            print("CUT AUDIO:", segment_audio)

            # ส่งเสียงที่ตัดแล้วเข้า Whisper
            result = transcribe_audio(
                segment_audio,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                initial_prompt=initial_prompt,
                post_process=post_process,
            )

            if hasattr(result, "segments"):
                whisper_segments = result.segments
            elif isinstance(result, dict):
                whisper_segments = result.get("segments", [])
            else:
                whisper_segments = []

            print(
                "WHISPER SEGMENTS:",
                len(whisper_segments),
            )

            for whisper_segment in whisper_segments:

                if hasattr(whisper_segment, "text"):
                    text = whisper_segment.text.strip()
                    local_start = float(whisper_segment.start)
                    local_end = float(whisper_segment.end)

                else:
                    text = whisper_segment.get(
                        "text",
                        "",
                    ).strip()

                    local_start = float(
                        whisper_segment.get(
                            "start",
                            0,
                        )
                    )

                    local_end = float(
                        whisper_segment.get(
                            "end",
                            local_start,
                        )
                    )

                if not text:
                    continue

                results.append(
                    {
                        "start": round(
                            start + local_start,
                            3,
                        ),
                        "end": round(
                            start + local_end,
                            3,
                        ),
                        "speaker": speaker,
                        "text": text,
                    }
                )

        except Exception as exc:

            print(
                f"ERROR TRANSCRIBE "
                f"{speaker} "
                f"{start:.3f}-{end:.3f}: "
                f"{type(exc).__name__}: {exc}"
            )

            raise

        finally:

            if (
                segment_audio
                and segment_audio.exists()
            ):
                segment_audio.unlink(
                    missing_ok=True
                )

        if progress_callback:

            progress = int(
                ((index + 1) / total) * 100
            )

            progress_callback(progress)

    results.sort(
        key=lambda x: x["start"]
    )

    print(
        "========== TRANSCRIBE BY SPEAKER COMPLETED =========="
    )

    print(
        "TOTAL RESULTS:",
        len(results),
    )

    return results