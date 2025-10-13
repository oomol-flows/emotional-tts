"""
Audio speed adjustment utility for reference audio preprocessing.
Intelligently adjusts speech rate while preserving audio quality and timbre.
"""

import os
import tempfile
import numpy as np
from typing import Tuple, Optional

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    raise ImportError(
        "Required libraries not found. Please ensure librosa and soundfile are installed. "
        f"Original error: {e}"
    )


def analyze_speech_rate(audio_path: str, sr: int = 22050) -> Tuple[float, float]:
    """
    Analyze speech rate of an audio file.

    Args:
        audio_path: Path to audio file
        sr: Target sample rate for analysis

    Returns:
        Tuple of (tempo_bpm, speech_rate_ratio)
        - tempo_bpm: Estimated tempo in beats per minute
        - speech_rate_ratio: Relative speech rate (1.0 = normal, <1.0 = slow, >1.0 = fast)
    """
    # Load audio
    y, sr_orig = librosa.load(audio_path, sr=sr, mono=True)

    # Remove silence for more accurate analysis
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)

    if len(y_trimmed) < sr * 0.5:  # Less than 0.5 seconds of audio
        # Too short to analyze, assume normal speed
        return 120.0, 1.0

    # Estimate tempo
    onset_env = librosa.onset.onset_strength(y=y_trimmed, sr=sr)
    try:
        # Try new API first (librosa >= 0.10.0)
        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
    except AttributeError:
        # Fall back to old API
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    # Calculate speech rate based on onset density
    # More onsets = faster speech
    onset_frames = librosa.onset.onset_detect(y=y_trimmed, sr=sr, units='frames')
    duration = len(y_trimmed) / sr
    onset_rate = len(onset_frames) / duration if duration > 0 else 0

    # Normal speech has roughly 3-5 onsets per second
    # Adjust this based on typical speech patterns
    normal_onset_rate = 4.0
    speech_rate_ratio = onset_rate / normal_onset_rate if normal_onset_rate > 0 else 1.0

    return tempo, speech_rate_ratio


def calculate_optimal_speed_factor(
    speech_rate_ratio: float,
    min_threshold: float = 0.75,
    max_threshold: float = 1.25,
    target_rate: float = 1.0
) -> float:
    """
    Calculate optimal speed adjustment factor.

    Args:
        speech_rate_ratio: Current speech rate ratio (1.0 = normal)
        min_threshold: Minimum ratio to trigger speed up (e.g., 0.75 = 75% of normal)
        max_threshold: Maximum ratio to trigger slow down (e.g., 1.25 = 125% of normal)
        target_rate: Target speech rate ratio

    Returns:
        Speed adjustment factor (1.0 = no change)
    """
    if speech_rate_ratio < min_threshold:
        # Speech is too slow, speed up
        # Target: bring it closer to normal (target_rate)
        speed_factor = target_rate / speech_rate_ratio
        # Limit maximum speedup to avoid artifacts
        speed_factor = min(speed_factor, 1.3)
        return speed_factor

    elif speech_rate_ratio > max_threshold:
        # Speech is too fast, slow down
        # Target: bring it closer to normal (target_rate)
        speed_factor = target_rate / speech_rate_ratio
        # Limit maximum slowdown to avoid artifacts
        speed_factor = max(speed_factor, 0.7)
        return speed_factor

    else:
        # Speech rate is acceptable, no adjustment needed
        return 1.0


def adjust_audio_speed(
    audio_path: str,
    speed_factor: float,
    output_path: Optional[str] = None,
    sr: int = 22050
) -> str:
    """
    Adjust audio speed using high-quality time stretching.
    Preserves pitch and timbre.

    Args:
        audio_path: Path to input audio file
        speed_factor: Speed factor (>1.0 = faster, <1.0 = slower, 1.0 = no change)
        output_path: Path to save adjusted audio (optional, creates temp file if None)
        sr: Sample rate

    Returns:
        Path to the adjusted audio file
    """
    if speed_factor == 1.0:
        # No adjustment needed
        return audio_path

    # Load audio at original sample rate to preserve quality
    y, sr_orig = librosa.load(audio_path, sr=None, mono=True)

    # Apply time stretching (preserves pitch)
    # rate = 1/speed_factor because librosa.effects.time_stretch uses rate parameter
    # rate > 1.0 = slower, rate < 1.0 = faster
    rate = 1.0 / speed_factor
    y_stretched = librosa.effects.time_stretch(y, rate=rate)

    # Create output path if not provided
    if output_path is None:
        temp_fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='speed_adjusted_')
        os.close(temp_fd)

    # Save adjusted audio
    sf.write(output_path, y_stretched, sr_orig)

    return output_path


def auto_adjust_reference_audio(
    audio_path: str,
    output_path: Optional[str] = None,
    min_threshold: float = 0.75,
    max_threshold: float = 1.25,
    target_rate: float = 1.0,
    verbose: bool = True
) -> Tuple[str, dict]:
    """
    Automatically analyze and adjust reference audio speed if needed.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save adjusted audio (optional)
        min_threshold: Minimum speech rate ratio to trigger adjustment
        max_threshold: Maximum speech rate ratio to trigger adjustment
        target_rate: Target speech rate ratio (1.0 = normal)
        verbose: Print analysis information

    Returns:
        Tuple of (adjusted_audio_path, metadata_dict)
        - adjusted_audio_path: Path to adjusted audio (same as input if no adjustment)
        - metadata_dict: Dictionary with analysis and adjustment info
    """
    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")

    # Analyze speech rate
    tempo, speech_rate_ratio = analyze_speech_rate(audio_path)

    # Calculate optimal speed factor
    speed_factor = calculate_optimal_speed_factor(
        speech_rate_ratio,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        target_rate=target_rate
    )

    metadata = {
        'original_path': audio_path,
        'tempo_bpm': tempo,
        'speech_rate_ratio': speech_rate_ratio,
        'speed_factor': speed_factor,
        'adjusted': speed_factor != 1.0
    }

    if verbose:
        print(f">> Audio speed analysis:")
        print(f"   Tempo: {tempo:.1f} BPM")
        print(f"   Speech rate ratio: {speech_rate_ratio:.2f}x")

        if speed_factor == 1.0:
            print(f"   ✓ Speech rate is acceptable, no adjustment needed")
        elif speed_factor > 1.0:
            print(f"   → Speech is too slow, speeding up by {speed_factor:.2f}x")
        else:
            print(f"   → Speech is too fast, slowing down to {speed_factor:.2f}x")

    # Apply adjustment if needed
    if speed_factor != 1.0:
        adjusted_path = adjust_audio_speed(audio_path, speed_factor, output_path)
        metadata['adjusted_path'] = adjusted_path

        if verbose:
            print(f"   ✓ Adjusted audio saved to: {adjusted_path}")

        return adjusted_path, metadata
    else:
        metadata['adjusted_path'] = audio_path
        return audio_path, metadata
