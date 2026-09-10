import app.ffmpeg_config
import app.hf_patch
from collections.abc import Callable
from pathlib import Path
import traceback
import concurrent.futures

from app.config import get_settings
from app.postprocess import post_process_transcript, apply_dictionary, post_process_segments
from app.schemas import SegmentOut, TranscriptionOut

from app.diarization_service import diarize_audio
from app.whisper_service import get_model, transcribe_with_cloud_api
from app.alignment_service import find_best_speaker, merge_same_speaker_segments, renumber_speakers_chronologically


def align_whisper_segments_with_speakers(
    whisper_segments: list,
    speaker_segments: list[dict],
) -> list[dict]:
    """
    Align full-file Whisper segments with Pyannote speaker diarization using word-level timestamps.
    Splits segments when speaker changes so Doctor and Patient never get merged into one block.
    """
    if not speaker_segments:
        return [
            {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "speaker": "SPEAKER_00",
                "text": seg.text.strip(),
            }
            for seg in whisper_segments if seg.text.strip()
        ]

    aligned_results = []

    for seg in whisper_segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue

        words = getattr(seg, "words", None)
        if words:
            current_spk = None
            current_words = []
            current_start = None
            current_end = None

            for w in words:
                w_str = w.word
                if not w_str.strip():
                    if current_words:
                        current_words.append(w_str)
                    continue

                w_start = float(w.start)
                w_end = float(w.end)
                spk, _ = find_best_speaker(w_start, w_end, speaker_segments)
                if not spk:
                    spk = "SPEAKER_00"

                if current_spk is None:
                    current_spk = spk
                    current_start = w_start
                    current_end = w_end
                    current_words = [w_str]
                elif spk == current_spk:
                    current_end = w_end
                    current_words.append(w_str)
                else:
                    block_text = "".join(current_words).strip()
                    if block_text:
                        aligned_results.append({
                            "start": round(current_start, 3),
                            "end": round(current_end, 3),
                            "speaker": current_spk,
                            "text": block_text,
                        })
                    current_spk = spk
                    current_start = w_start
                    current_end = w_end
                    current_words = [w_str]

            if current_words:
                block_text = "".join(current_words).strip()
                if block_text:
                    aligned_results.append({
                        "start": round(current_start, 3),
                        "end": round(current_end, 3),
                        "speaker": current_spk,
                        "text": block_text,
                    })
        else:
            seg_start = float(seg.start)
            seg_end = float(seg.end)
            best_spk, _ = find_best_speaker(seg_start, seg_end, speaker_segments)
            aligned_results.append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "speaker": best_spk or "SPEAKER_00",
                "text": seg_text,
            })

    return aligned_results


def run_whisperx_pipeline(
    audio_path: Path,
    *,
    language: str | None = "th",
    task: str = "transcribe",
    beam_size: int | None = None,
    vad_filter: bool | None = None,
    initial_prompt: str | None = None,
    post_process: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    """
    Original Pristine Whole-File Transcription with Zero Hallucination & Smart Speaker Alignment.
    """
    settings = get_settings()
    audio_path = Path(audio_path)

    if progress_callback:
        progress_callback(10, "กำลังเตรียมไฟล์เสียง")

    print("========== FULL-FILE TRANSCRIPTION & DIARIZATION START ==========")
    print("Audio:", audio_path)

    # Run Speaker Diarization and Faster-Whisper in parallel
    if progress_callback:
        progress_callback(20, "กำลังแปลงเสียงและแยกผู้พูดพร้อมกัน")

    # Configure PyTorch CPU threads to avoid CPU thrashing between Pyannote and Whisper
    if settings.device == "cpu":
        import torch
        import os
        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(max(2, cpu_cores // 2))

    resolved_beam = beam_size or settings.default_beam_size
    resolved_prompt = initial_prompt or settings.initial_prompt
    resolved_vad = vad_filter if vad_filter is not None else settings.default_vad_filter

    def do_whisper_transcribe():
        if settings.groq_api_key or settings.transcription_provider in {"groq", "openai"}:
            print(f"Cloud transcription starting (provider={settings.transcription_provider})...")
            return transcribe_with_cloud_api(
                audio_path,
                language=language or "th",
                initial_prompt=resolved_prompt,
            )
        else:
            print(f"Local Whisper transcribing: model={settings.model_size}, beam_size={resolved_beam}, vad={resolved_vad}, prompt={bool(resolved_prompt)}...")
            model = get_model()
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language or "th",
                task=task,
                beam_size=resolved_beam,
                vad_filter=resolved_vad,
                vad_parameters=dict(threshold=0.35, min_silence_duration_ms=400, speech_pad_ms=250),
                initial_prompt=resolved_prompt if resolved_prompt else None,
                condition_on_previous_text=settings.condition_on_previous_text,
                repetition_penalty=settings.repetition_penalty,
                no_repeat_ngram_size=settings.no_repeat_ngram_size,
                temperature=[0.0, 0.2],
                word_timestamps=True,
            )
            segs = [s for s in segments_iter if s.text.strip()]
            return segs, getattr(info, "duration", None), getattr(info, "language", language or "th")

    resolved_num_speakers = num_speakers if num_speakers is not None else settings.num_speakers
    resolved_min_speakers = min_speakers if min_speakers is not None else settings.min_speakers
    resolved_max_speakers = max_speakers if max_speakers is not None else settings.max_speakers

    def do_diarize():
        if settings.transcription_provider == "groq" or not settings.huggingface_token:
            print("Using ultra-fast cloud pipeline: AI Role classification enabled (skipping slow CPU Pyannote)...")
            return []
        print(f"Pyannote diarization starting (num_speakers={resolved_num_speakers}, min={resolved_min_speakers}, max={resolved_max_speakers})...")
        return diarize_audio(
            audio_path,
            num_speakers=resolved_num_speakers,
            min_speakers=resolved_min_speakers,
            max_speakers=resolved_max_speakers,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_diarize = executor.submit(do_diarize)
        future_whisper = executor.submit(do_whisper_transcribe)

        speaker_segments = future_diarize.result()
        whisper_segments, duration, detected_lang = future_whisper.result()

    print(f"Parallel processing completed: {len(whisper_segments)} whisper segments, {len(speaker_segments)} diarization segments.")

    # Step 3: Align Whisper Segments with Speakers
    if progress_callback:
        progress_callback(85, "กำลังจัดตำแหน่งผู้พูดกับข้อความ")

    final_segments = align_whisper_segments_with_speakers(whisper_segments, speaker_segments)

    # Step 4: Sort and merge adjacent same-speaker segments, normalize speaker order chronologically
    final_segments.sort(key=lambda x: x["start"])
    final_segments = renumber_speakers_chronologically(final_segments)
    final_segments = merge_same_speaker_segments(final_segments, max_gap=0.6)
    final_segments = renumber_speakers_chronologically(final_segments)

    # Step 5: Post-processing (Dictionary, Homoglyph Normalizer & AI Proofreading)
    if progress_callback:
        progress_callback(95, "กำลังตรวจแก้คำผิดและจัดรูปแบบ")

    formatted_segments = []
    for idx, seg in enumerate(final_segments):
        speaker_label = seg.get("speaker") or "SPEAKER_00"
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue

        formatted_segments.append(
            SegmentOut(
                id=idx,
                start=seg["start"],
                end=seg["end"],
                text=seg_text,
                speaker=speaker_label,
                role=None,
            )
        )

    resolved_post_process = post_process or settings.post_process_mode
    formatted_segments, full_final_text, full_raw_text, post_processed_by = post_process_segments(
        formatted_segments,
        resolved_post_process,
    )


    if progress_callback:
        progress_callback(100, "แปลงเสียงและแยกผู้พูดเสร็จแล้ว")

    print("========== TRANSCRIPTION COMPLETED ==========")
    print(f"Total output segments: {len(formatted_segments)}")

    return {
        "text": full_final_text,
        "raw_text": full_raw_text,
        "segments": [s.model_dump() for s in formatted_segments],
        "speakers": speaker_segments,
        "language": detected_lang,
        "language_probability": 1.0,
        "duration": duration,
        "post_processed_by": post_processed_by,
    }
