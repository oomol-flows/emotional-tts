#region generated meta
import typing
class Inputs(typing.TypedDict):
    audio_file: str
    auto_detect_vocal: bool
    start_time: float | None
    duration: float | None
class Outputs(typing.TypedDict):
    reference_audio: typing.NotRequired[str]
#endregion

from oocana import Context
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import tempfile
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model


def detect_vocal_segments(audio_path: str, context: Context) -> list:
    """
    Detect vocal segments in audio using Demucs for vocal separation
    and silence detection to find continuous vocal parts.

    Args:
        audio_path: Path to audio file
        context: OOMOL context object

    Returns:
        List of tuples (start_ms, end_ms) representing vocal segments
    """
    context.logger.info("Separating vocals from audio using Demucs...")

    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()

    try:
        # Load pretrained Demucs model (htdemucs for high quality)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        context.logger.info(f"Using device: {device}")

        model = get_model("htdemucs")
        model.to(device)
        model.eval()

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Ensure stereo (Demucs expects stereo input)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        waveform = waveform.to(device)

        # Apply source separation
        context.logger.info("Running vocal separation...")
        with torch.no_grad():
            sources = apply_model(model, waveform.unsqueeze(0), device=device)[0]

        # Demucs outputs: [drums, bass, other, vocals]
        vocals = sources[3].cpu()

        # Save vocals to temporary file
        vocal_path = os.path.join(temp_dir, "vocals.wav")
        torchaudio.save(vocal_path, vocals, sample_rate)

        context.logger.info("Vocal separation completed, analyzing vocal segments...")

        # Load separated vocal track with pydub for silence detection
        vocal_audio = AudioSegment.from_wav(vocal_path)

        # Detect non-silent segments in vocal track
        # Parameters: min_silence_len (ms), silence_thresh (dBFS)
        nonsilent_segments = detect_nonsilent(
            vocal_audio,
            min_silence_len=500,  # Minimum 500ms silence to split segments
            silence_thresh=-40    # Audio below -40dBFS considered silence
        )

        context.logger.info(f"Found {len(nonsilent_segments)} vocal segments")

        return nonsilent_segments

    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def find_best_vocal_segment(segments: list, target_duration_ms: int, context: Context) -> tuple:
    """
    Find the best vocal segment that matches target duration.

    Args:
        segments: List of (start_ms, end_ms) tuples
        target_duration_ms: Target duration in milliseconds
        context: OOMOL context object

    Returns:
        Tuple of (start_ms, end_ms) for the best segment
    """
    if not segments:
        raise ValueError("No vocal segments found in audio")

    # Find the longest continuous vocal segment
    longest_segment = max(segments, key=lambda s: s[1] - s[0])
    segment_duration = longest_segment[1] - longest_segment[0]

    context.logger.info(
        f"Longest vocal segment: {segment_duration/1000:.2f}s "
        f"({longest_segment[0]/1000:.2f}s - {longest_segment[1]/1000:.2f}s)"
    )

    # If the longest segment is shorter than target, use the whole segment
    if segment_duration <= target_duration_ms:
        context.logger.info(f"Using entire segment ({segment_duration/1000:.2f}s)")
        return longest_segment

    # If longer than target, extract from the middle of the segment
    start_ms = longest_segment[0] + (segment_duration - target_duration_ms) // 2
    end_ms = start_ms + target_duration_ms

    context.logger.info(
        f"Extracting {target_duration_ms/1000}s from middle of segment "
        f"({start_ms/1000:.2f}s - {end_ms/1000:.2f}s)"
    )

    return (start_ms, end_ms)


def main(params: Inputs, context: Context) -> Outputs:
    """
    Extract vocal reference audio with automatic vocal detection.

    Args:
        params: Input parameters containing audio file path and options
        context: OOMOL context object

    Returns:
        Output dictionary with extracted reference audio file path
    """
    audio_file = params["audio_file"]
    auto_detect = params.get("auto_detect_vocal", True)
    duration = params.get("duration") or 25

    # Validate input file
    if not os.path.exists(audio_file):
        raise ValueError(f"Audio file not found: {audio_file}")

    # Load audio file
    context.logger.info(f"Loading audio file: {audio_file}")
    audio = AudioSegment.from_file(audio_file)
    audio_duration_ms = len(audio)

    # Determine extraction range
    if auto_detect:
        # Auto-detect vocal segments
        context.logger.info("Auto-detecting vocal segments...")
        vocal_segments = detect_vocal_segments(audio_file, context)
        start_ms, end_ms = find_best_vocal_segment(
            vocal_segments,
            duration * 1000,
            context
        )
    else:
        # Manual time range
        start_time = params.get("start_time") or 0
        start_ms = int(start_time * 1000)
        end_ms = int((start_time + duration) * 1000)

        # Validate time range
        if start_ms >= audio_duration_ms:
            raise ValueError(f"Start time {start_time}s exceeds audio duration {audio_duration_ms/1000}s")

        # Adjust end time if exceeds audio duration
        if end_ms > audio_duration_ms:
            context.logger.warning(
                f"Requested end time {end_ms/1000}s exceeds audio duration {audio_duration_ms/1000}s, "
                f"adjusting to audio end"
            )
            end_ms = audio_duration_ms

    # Extract audio segment
    context.logger.info(f"Extracting audio from {start_ms/1000:.2f}s to {end_ms/1000:.2f}s")
    extracted_audio = audio[start_ms:end_ms]

    # Generate output file path
    output_dir = "/oomol-driver/oomol-storage/audio-references"
    os.makedirs(output_dir, exist_ok=True)

    input_basename = os.path.splitext(os.path.basename(audio_file))[0]
    if auto_detect:
        output_filename = f"{input_basename}_vocal_ref_{(end_ms-start_ms)/1000:.1f}s.wav"
    else:
        output_filename = f"{input_basename}_ref_{start_ms/1000:.1f}s_{(end_ms-start_ms)/1000:.1f}s.wav"
    output_path = os.path.join(output_dir, output_filename)

    # Export extracted audio
    context.logger.info(f"Saving reference audio to: {output_path}")
    extracted_audio.export(output_path, format="wav")

    context.logger.info(f"Successfully extracted {(end_ms - start_ms)/1000:.2f}s reference audio")

    return {"reference_audio": output_path}
