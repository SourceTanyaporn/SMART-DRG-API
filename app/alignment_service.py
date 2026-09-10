def get_segment_value(
    segment,
    key: str,
    default=None,
):
    """
    รองรับทั้ง dict และ Pydantic model เช่น SegmentOut
    """

    if isinstance(segment, dict):
        return segment.get(key, default)

    return getattr(
        segment,
        key,
        default,
    )


# def calculate_overlap(
#     start_a: float,
#     end_a: float,
#     start_b: float,
#     end_b: float,
# ) -> float:

#     start = max(
#         start_a,
#         start_b,
#     )

#     end = min(
#         end_a,
#         end_b,
#     )

#     return max(
#         0.0,
#         end - start,
#     )
def calculate_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:

    start = max(start_a, start_b)
    end = min(end_a, end_b)

    return max(0.0, end - start)

def find_best_speaker(
    start: float,
    end: float,
    speaker_segments: list,
):
    speaker_overlap = {}

    for speaker_segment in speaker_segments:

        overlap = calculate_overlap(
            start,
            end,
            float(speaker_segment["start"]),
            float(speaker_segment["end"]),
        )

        if overlap <= 0:
            continue

        speaker = speaker_segment["speaker"]

        speaker_overlap[speaker] = (
            speaker_overlap.get(speaker, 0.0)
            + overlap
        )

    if not speaker_overlap:
        if not speaker_segments:
            return None, 0.0
        # Fallback to closest speaker segment in time
        mid = (start + end) / 2.0
        closest = min(
            speaker_segments,
            key=lambda s: min(abs(mid - float(s["start"])), abs(mid - float(s["end"])))
        )
        return closest.get("speaker"), 0.0

    best_speaker = max(
        speaker_overlap,
        key=speaker_overlap.get,
    )

    return (
        best_speaker,
        speaker_overlap[best_speaker],
    )


def renumber_speakers_chronologically(segments: list) -> list:
    """
    Ensure the first person who speaks is ALWAYS SPEAKER_00 (ผู้พูดคนที่ 1),
    the second person is SPEAKER_01 (ผู้พูดคนที่ 2), etc.
    This prevents arbitrary Pyannote cluster numbering from swapping speakers in UI.
    """
    if not segments:
        return []

    speaker_map = {}
    next_idx = 0

    for seg in segments:
        if isinstance(seg, dict):
            raw_spk = seg.get("speaker")
        else:
            raw_spk = getattr(seg, "speaker", None)

        if not raw_spk:
            continue

        if raw_spk not in speaker_map:
            speaker_map[raw_spk] = f"SPEAKER_{next_idx:02d}"
            next_idx += 1

        if isinstance(seg, dict):
            seg["speaker"] = speaker_map[raw_spk]
        else:
            seg.speaker = speaker_map[raw_spk]

    return segments


def align_transcript_with_speakers(
    transcript_segments: list,
    speaker_segments: list,
):
    result = []

    for segment in transcript_segments:

        if hasattr(segment, "start"):
            start = float(segment.start)
            end = float(segment.end)
            text = segment.text.strip()
        else:
            start = float(segment.get("start", 0))
            end = float(
                segment.get("end", start)
            )
            text = segment.get(
                "text",
                "",
            ).strip()

        if not text:
            continue

        speaker, overlap = find_best_speaker(
            start,
            end,
            speaker_segments,
        )

        result.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "text": text,
                "speaker_overlap": round(
                    overlap,
                    3,
                ),
            }
        )

    return result

# def find_best_speaker(
#     start: float,
#     end: float,
#     speaker_segments: list,
# ):
#     candidates = []

#     for speaker_segment in speaker_segments:

#         speaker_start = float(
#             get_segment_value(
#                 speaker_segment,
#                 "start",
#                 0,
#             )
#         )

#         speaker_end = float(
#             get_segment_value(
#                 speaker_segment,
#                 "end",
#                 speaker_start,
#             )
#         )

#         speaker = get_segment_value(
#             speaker_segment,
#             "speaker",
#             None,
#         )

#         overlap = calculate_overlap(
#             start,
#             end,
#             speaker_start,
#             speaker_end,
#         )

#         if overlap <= 0:
#             continue

#         candidates.append(
#             {
#                 "speaker": speaker,
#                 "start": speaker_start,
#                 "end": speaker_end,
#                 "overlap": overlap,
#             }
#         )

#     if not candidates:
#         return None, 0.0

#     # ------------------------------------------------
#     # 1. ถ้ามี speaker segment ที่อยู่ "ข้างใน"
#     #    transcript segment ให้ priority ก่อน
#     #
#     #    เช่น
#     #    Whisper     14 - 16
#     #    Speaker 01 15 - 15.47
#     #
#     #    ให้ถือว่า speaker 01 มีความสำคัญ
#     # ------------------------------------------------

#     nested_candidates = []

#     for candidate in candidates:

#         speaker_start = candidate["start"]
#         speaker_end = candidate["end"]

#         if (
#             speaker_start >= start
#             and speaker_end <= end
#         ):
#             nested_candidates.append(
#                 candidate
#             )

#     if nested_candidates:

#         best = max(
#             nested_candidates,
#             key=lambda x: x["overlap"],
#         )

#         return (
#             best["speaker"],
#             best["overlap"],
#         )

#     # ------------------------------------------------
#     # 2. ถ้าไม่มี nested speaker
#     #    ใช้ overlap มากที่สุด
#     # ------------------------------------------------

#     best = max(
#         candidates,
#         key=lambda x: x["overlap"],
#     )

#     return (
#         best["speaker"],
#         best["overlap"],
#     )


# def align_transcript_with_speakers(
#     transcript_segments: list,
#     speaker_segments: list,
# ):
#     result = []

#     for segment in transcript_segments:

#         start = float(
#             get_segment_value(
#                 segment,
#                 "start",
#                 0,
#             )
#         )

#         end = float(
#             get_segment_value(
#                 segment,
#                 "end",
#                 start,
#             )
#         )

#         text = str(
#             get_segment_value(
#                 segment,
#                 "text",
#                 "",
#             )
#             or ""
#         ).strip()

#         if not text:
#             continue

#         speaker, overlap = find_best_speaker(
#             start,
#             end,
#             speaker_segments,
#         )

#         result.append(
#             {
#                 "start": start,
#                 "end": end,
#                 "speaker": speaker,
#                 "text": text,
#                 "speaker_overlap": round(
#                     overlap,
#                     3,
#                 ),
#             }
#         )

#     return result

def merge_same_speaker_segments(
    segments: list,
    max_gap: float = 0.5,
):
    if not segments:
        return []

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

        speaker = segment.get(
            "speaker"
        )

        text = (
            segment.get("text", "")
            .strip()
        )

        if end <= start:
            continue

        if not text:
            continue

        current = (
            merged[-1]
            if merged
            else None
        )

        if current is None:

            merged.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                }
            )

            continue

        gap = (
            start
            - float(current["end"])
        )

        same_speaker = (
            current["speaker"]
            == speaker
        )

        if (
            same_speaker
            and gap <= max_gap
        ):

            current["end"] = max(
                float(current["end"]),
                end,
            )

            current["text"] = (
                current["text"].rstrip()
                + " "
                + text.lstrip()
            )

        else:

            merged.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                }
            )

    return merged