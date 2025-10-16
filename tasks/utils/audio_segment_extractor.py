"""
Audio segment extraction utility for SRT-based reference audio.
Extracts time-aligned audio segments from full audio file based on SRT timestamps.
"""

import os
import tempfile
from typing import Tuple, Optional

try:
    from pydub import AudioSegment
except ImportError as e:
    raise ImportError(
        "Required library pydub not found. Please ensure pydub is installed. "
        f"Original error: {e}"
    )


def extract_audio_segment(
    full_audio_path: str,
    start_time_sec: float,
    end_time_sec: float,
    offset_ms: int = 0,
    output_path: Optional[str] = None,
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 10.0,
    padding_ms: int = 100
) -> str:
    """
    Extract a time-aligned audio segment from full audio file.

    Args:
        full_audio_path: Path to the full audio file
        start_time_sec: Start time in seconds
        end_time_sec: End time in seconds
        offset_ms: Time offset adjustment in milliseconds (positive = delay, negative = advance)
        output_path: Path to save extracted segment (auto-generated if None)
        min_duration_sec: Minimum segment duration (pad if shorter)
        max_duration_sec: Maximum segment duration (trim if longer)
        padding_ms: Add padding at edges for smoother reference (default 100ms)

    Returns:
        Path to extracted audio segment file
    """
    if not os.path.exists(full_audio_path):
        raise ValueError(f"Full audio file not found: {full_audio_path}")

    # Load full audio
    try:
        full_audio = AudioSegment.from_file(full_audio_path)
    except Exception as e:
        raise ValueError(f"Failed to load audio file {full_audio_path}: {str(e)}")

    full_duration_ms = len(full_audio)

    # Convert times to milliseconds
    start_ms = int(start_time_sec * 1000)
    end_ms = int(end_time_sec * 1000)
    duration_ms = end_ms - start_ms

    # Apply offset
    start_ms += offset_ms
    end_ms += offset_ms

    # Add padding for smoother edges
    start_ms = max(0, start_ms - padding_ms)
    end_ms = min(full_duration_ms, end_ms + padding_ms)

    # Enforce duration constraints
    duration_ms = end_ms - start_ms
    min_duration_ms = int(min_duration_sec * 1000)
    max_duration_ms = int(max_duration_sec * 1000)

    if duration_ms < min_duration_ms:
        # Segment too short, extend symmetrically
        needed_ms = min_duration_ms - duration_ms
        extend_start = needed_ms // 2
        extend_end = needed_ms - extend_start

        start_ms = max(0, start_ms - extend_start)
        end_ms = min(full_duration_ms, end_ms + extend_end)

        # If still too short after extending, duplicate audio
        duration_ms = end_ms - start_ms
        if duration_ms < min_duration_ms:
            segment = full_audio[start_ms:end_ms]
            # Loop audio to meet minimum duration
            loops_needed = (min_duration_ms // duration_ms) + 1
            segment = segment * loops_needed
            segment = segment[:min_duration_ms]
        else:
            segment = full_audio[start_ms:end_ms]

    elif duration_ms > max_duration_ms:
        # Segment too long, trim from center
        center_ms = (start_ms + end_ms) // 2
        half_max = max_duration_ms // 2
        start_ms = max(0, center_ms - half_max)
        end_ms = min(full_duration_ms, center_ms + half_max)
        segment = full_audio[start_ms:end_ms]

    else:
        # Duration is acceptable
        segment = full_audio[start_ms:end_ms]

    # Add fade in/out for smoother reference
    fade_duration = min(50, len(segment) // 10)  # 50ms or 10% of duration
    if fade_duration > 0 and len(segment) > fade_duration * 2:
        segment = segment.fade_in(fade_duration).fade_out(fade_duration)

    # Create output path if not provided
    if output_path is None:
        temp_fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='ref_segment_')
        os.close(temp_fd)

    # Export segment
    try:
        segment.export(output_path, format='wav')
    except Exception as e:
        raise ValueError(f"Failed to export audio segment: {str(e)}")

    return output_path


def validate_full_audio_duration(
    full_audio_path: str,
    srt_max_time_sec: float,
    tolerance_sec: float = 5.0
) -> Tuple[bool, str]:
    """
    Validate that full audio duration matches SRT file duration.

    Args:
        full_audio_path: Path to full audio file
        srt_max_time_sec: Maximum timestamp in SRT file
        tolerance_sec: Allowed time difference in seconds

    Returns:
        Tuple of (is_valid, message)
    """
    if not os.path.exists(full_audio_path):
        return False, f"Audio file not found: {full_audio_path}"

    try:
        audio = AudioSegment.from_file(full_audio_path)
        audio_duration_sec = len(audio) / 1000.0

        time_diff = abs(audio_duration_sec - srt_max_time_sec)

        if time_diff <= tolerance_sec:
            return True, f"Audio duration ({audio_duration_sec:.2f}s) matches SRT max time ({srt_max_time_sec:.2f}s)"
        else:
            return False, f"Audio duration ({audio_duration_sec:.2f}s) differs from SRT max time ({srt_max_time_sec:.2f}s) by {time_diff:.2f}s"

    except Exception as e:
        return False, f"Failed to validate audio: {str(e)}"


def analyze_segment_quality(
    segment_path: str,
    min_rms_threshold: float = 0.01,
    max_silence_ratio: float = 0.5
) -> Tuple[bool, str]:
    """
    Analyze if extracted segment has sufficient audio quality for reference.

    Args:
        segment_path: Path to audio segment
        min_rms_threshold: Minimum RMS energy threshold
        max_silence_ratio: Maximum allowed silence ratio

    Returns:
        Tuple of (is_good_quality, message)
    """
    if not os.path.exists(segment_path):
        return False, "Segment file not found"

    try:
        import numpy as np
        from pydub import AudioSegment

        audio = AudioSegment.from_file(segment_path)

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Normalize
        if audio.sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            samples = samples.astype(np.float32) / 2147483648.0

        # Handle stereo
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = np.mean(samples, axis=1)

        # Calculate RMS
        rms = np.sqrt(np.mean(samples ** 2))

        if rms < min_rms_threshold:
            return False, f"Segment too quiet (RMS: {rms:.4f})"

        # Calculate silence ratio
        silence_threshold = min_rms_threshold * 2
        silence_samples = np.sum(np.abs(samples) < silence_threshold)
        silence_ratio = silence_samples / len(samples)

        if silence_ratio > max_silence_ratio:
            return False, f"Too much silence ({silence_ratio * 100:.1f}%)"

        return True, f"Good quality (RMS: {rms:.4f}, Silence: {silence_ratio * 100:.1f}%)"

    except Exception as e:
        return False, f"Failed to analyze segment: {str(e)}"
