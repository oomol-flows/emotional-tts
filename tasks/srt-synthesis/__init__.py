#region generated meta
import typing
class Inputs(typing.TypedDict):
    srt_file: str
    spk_audio_prompt: str
    emo_control_mode: typing.Literal["speaker", "reference", "vector", "text"] | None
    emo_audio_prompt: str | None
    emo_weight: float | None
    emo_text: str | None
    emo_vec_happy: float | None
    emo_vec_angry: float | None
    emo_vec_sad: float | None
    emo_vec_afraid: float | None
    emo_vec_disgusted: float | None
    emo_vec_melancholic: float | None
    emo_vec_surprised: float | None
    emo_vec_calm: float | None
    use_random: bool | None
    max_mel_tokens: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_text_tokens_per_segment: int | None
    speed_factor: float | None
    time_sync_mode: typing.Literal["stretch", "crop", "overlay"] | None
class Outputs(typing.TypedDict):
    audio: typing.NotRequired[str]
    srt_with_metadata: typing.NotRequired[str]
#endregion

import os
import sys
import time
import tempfile
from oocana import Context

# Suppress warnings before importing
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import model downloader utility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_downloader import ensure_model_downloaded

# Set HuggingFace cache directory
os.environ['HF_HUB_CACHE'] = '/oomol-driver/oomol-storage/indextts-checkpoints/hf_cache'

# Import required libraries
try:
    import pysrt
    from pydub import AudioSegment
except ImportError as e:
    raise ImportError(
        "Required libraries not found. Please ensure pysrt and pydub are installed. "
        f"Original error: {e}"
    )

# Import IndexTTS2 from installed package (not local directory)
# The local 'indextts' task folder has the same name as the installed package,
# so we need to manipulate sys.path to ensure Python imports the installed package
_original_sys_path = sys.path.copy()
_tasks_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Remove all paths that could lead to the tasks directory
_paths_to_remove = [p for p in sys.path if _tasks_dir in p or p == _tasks_dir]
for p in _paths_to_remove:
    sys.path.remove(p)

try:
    from indextts.infer_v2 import IndexTTS2
except ImportError as e:
    raise ImportError(
        "IndexTTS package not found. Please ensure the bootstrap script has run successfully. "
        f"Original error: {e}"
    )
finally:
    # Restore sys.path
    sys.path = _original_sys_path

# Global model instance (loaded once per task lifecycle)
_tts_model = None

def get_tts_model(model_dir: str, cfg_path: str):
    """
    Get or initialize the IndexTTS2 model (singleton pattern)
    Ensures model is downloaded on first execution.
    """
    global _tts_model

    if _tts_model is None:
        # Ensure model is downloaded before initialization
        ensure_model_downloaded(model_dir=model_dir)

        print(">> Initializing IndexTTS2 model...")
        _tts_model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=True,  # Enable FP16 for faster inference
            use_cuda_kernel=True,  # Enable CUDA kernel for BigVGAN
            use_deepspeed=False  # DeepSpeed disabled by default
        )
        print(">> IndexTTS2 model initialized successfully")

    return _tts_model

def parse_srt_timestamp(timestamp_str: str) -> float:
    """
    Parse SRT timestamp string to seconds
    Format: HH:MM:SS,mmm
    """
    hours, minutes, rest = timestamp_str.split(':')
    seconds, milliseconds = rest.split(',')

    total_seconds = (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(milliseconds) / 1000.0
    )

    return total_seconds

def synthesize_subtitle_entry(
    tts,
    text: str,
    spk_audio_prompt: str,
    emo_control_mode: str,
    emo_audio_prompt: str | None,
    emo_weight: float,
    emo_text: str | None,
    emo_vector: list,
    use_random: bool,
    max_mel_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_text_tokens_per_segment: int,
    temp_dir: str,
    target_duration_sec: float | None = None
) -> str:
    """
    Synthesize audio for a single subtitle entry
    Returns the path to the generated audio file

    Args:
        target_duration_sec: Expected duration in seconds for speed estimation (unused for now to avoid model conflicts)
    """
    # Generate unique output path
    output_path = os.path.join(temp_dir, f"segment_{int(time.time() * 1000000)}.wav")

    # Note: We pass target_duration_sec for future use but don't modify max_mel_tokens
    # to avoid tensor dimension mismatches in the TTS model

    # Prepare inference parameters based on emotion control mode
    use_emo_text = False
    final_emo_audio = None
    final_emo_vector = None

    if emo_control_mode == "speaker":
        final_emo_audio = None
        final_emo_vector = None
    elif emo_control_mode == "reference":
        if emo_audio_prompt and os.path.exists(emo_audio_prompt):
            final_emo_audio = emo_audio_prompt
        else:
            raise ValueError("Emotion reference audio is required for 'reference' mode")
        final_emo_vector = None
    elif emo_control_mode == "vector":
        final_emo_audio = None
        final_emo_vector = emo_vector
    elif emo_control_mode == "text":
        use_emo_text = True
        final_emo_audio = None
        final_emo_vector = None
    else:
        raise ValueError(f"Invalid emotion control mode: {emo_control_mode}")

    # Run inference
    result_path = tts.infer(
        spk_audio_prompt=spk_audio_prompt,
        text=text,
        output_path=output_path,
        emo_audio_prompt=final_emo_audio,
        emo_alpha=emo_weight,
        emo_vector=final_emo_vector,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=use_random,
        verbose=False,
        max_text_tokens_per_segment=max_text_tokens_per_segment,
        # Generation parameters
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        max_mel_tokens=max_mel_tokens,
        length_penalty=0.0,
        num_beams=3,
        repetition_penalty=10.0
    )

    return result_path

def main(params: Inputs, context: Context) -> Outputs:
    """
    SRT file text-to-speech synthesis task

    Args:
        params: Input parameters including SRT file, voice reference, and emotion controls
        context: OOMOL context object

    Returns:
        Output dictionary with generated audio file path and metadata SRT
    """
    # Model directory and config path
    model_dir = "/oomol-driver/oomol-storage/indextts-checkpoints"
    cfg_path = os.path.join(model_dir, "config.yaml")

    # Get TTS model (will download if needed)
    tts = get_tts_model(model_dir, cfg_path)

    # Extract parameters with defaults for nullable fields
    srt_file = params["srt_file"]
    spk_audio_prompt = params["spk_audio_prompt"]
    emo_control_mode = params.get("emo_control_mode") or "speaker"
    emo_audio_prompt = params.get("emo_audio_prompt")
    emo_weight = params.get("emo_weight") if params.get("emo_weight") is not None else 0.65
    emo_text = params.get("emo_text")
    use_random = params.get("use_random") if params.get("use_random") is not None else False
    max_mel_tokens = params.get("max_mel_tokens") if params.get("max_mel_tokens") is not None else 1500
    temperature = params.get("temperature") if params.get("temperature") is not None else 0.3
    top_p = params.get("top_p") if params.get("top_p") is not None else 0.9
    top_k = params.get("top_k") if params.get("top_k") is not None else 30
    max_text_tokens_per_segment = params.get("max_text_tokens_per_segment") if params.get("max_text_tokens_per_segment") is not None else 120
    speed_factor = params.get("speed_factor") if params.get("speed_factor") is not None else 1.0
    time_sync_mode = params.get("time_sync_mode") or "stretch"

    # Emotion vector from sliders (with defaults)
    emo_vector = [
        params.get("emo_vec_happy") if params.get("emo_vec_happy") is not None else 0.0,
        params.get("emo_vec_angry") if params.get("emo_vec_angry") is not None else 0.0,
        params.get("emo_vec_sad") if params.get("emo_vec_sad") is not None else 0.0,
        params.get("emo_vec_afraid") if params.get("emo_vec_afraid") is not None else 0.0,
        params.get("emo_vec_disgusted") if params.get("emo_vec_disgusted") is not None else 0.0,
        params.get("emo_vec_melancholic") if params.get("emo_vec_melancholic") is not None else 0.0,
        params.get("emo_vec_surprised") if params.get("emo_vec_surprised") is not None else 0.0,
        params.get("emo_vec_calm") if params.get("emo_vec_calm") is not None else 0.0
    ]

    # Validate SRT file
    if not srt_file or not os.path.exists(srt_file):
        raise ValueError(f"SRT file not found: {srt_file}")

    # Validate speaker audio prompt
    if not spk_audio_prompt or not os.path.exists(spk_audio_prompt):
        raise ValueError(f"Speaker audio prompt file not found: {spk_audio_prompt}")

    # Parse SRT file
    print(f">> Parsing SRT file: {srt_file}")
    try:
        subs = pysrt.open(srt_file, encoding='utf-8')
    except Exception as e:
        raise ValueError(f"Failed to parse SRT file: {str(e)}")

    if len(subs) == 0:
        raise ValueError("SRT file is empty or contains no valid subtitles")

    print(f">> Found {len(subs)} subtitle entries")

    # Create temporary directory for audio segments
    temp_dir = tempfile.mkdtemp(prefix="srt_synthesis_")

    try:
        # Generate audio for each subtitle entry
        audio_segments = []
        metadata_lines = []

        for idx, sub in enumerate(subs):
            print(f">> Processing subtitle {idx + 1}/{len(subs)}: {sub.text[:30]}...")

            # Clean subtitle text (remove formatting tags)
            text = sub.text_without_tags.strip()

            if not text:
                print(f"   Skipping empty subtitle {idx + 1}")
                continue

            # Calculate timestamps
            start_time_sec = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
            end_time_sec = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
            duration_sec = end_time_sec - start_time_sec

            # Synthesize audio for this subtitle
            audio_path = synthesize_subtitle_entry(
                tts=tts,
                text=text,
                spk_audio_prompt=spk_audio_prompt,
                emo_control_mode=emo_control_mode,
                emo_audio_prompt=emo_audio_prompt,
                emo_weight=emo_weight,
                emo_text=emo_text,
                emo_vector=emo_vector,
                use_random=use_random,
                max_mel_tokens=max_mel_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                temp_dir=temp_dir,
                target_duration_sec=duration_sec
            )

            # Load audio segment
            audio_segment = AudioSegment.from_wav(audio_path)

            # Apply speed factor if needed
            if speed_factor != 1.0:
                audio_segment = audio_segment.speedup(playback_speed=speed_factor)

            audio_duration_ms = len(audio_segment)

            # Store segment info
            audio_segments.append({
                'start_ms': start_time_sec * 1000,
                'end_ms': end_time_sec * 1000,
                'audio': audio_segment,
                'text': text,
                'index': idx + 1
            })

            # Add metadata for output SRT
            metadata_lines.append(f"{idx + 1}")
            metadata_lines.append(f"{sub.start} --> {sub.end}")
            metadata_lines.append(f"{text}")
            metadata_lines.append(f"# Audio duration: {audio_duration_ms}ms, Target duration: {duration_sec * 1000}ms")
            metadata_lines.append("")

            print(f"   Generated audio: {audio_duration_ms}ms (target: {duration_sec * 1000}ms)")

        if len(audio_segments) == 0:
            raise ValueError("No valid subtitles found to synthesize")

        # Merge audio segments with time synchronization
        print(f">> Merging audio segments (mode: {time_sync_mode})...")

        # Get the final end time
        final_end_ms = audio_segments[-1]['end_ms']

        # Create silent base track
        final_audio = AudioSegment.silent(duration=int(final_end_ms))

        # Define speed bounds for natural sounding speech
        MIN_SPEED_RATIO = 0.75  # Don't slow down more than 25%
        MAX_SPEED_RATIO = 1.35  # Don't speed up more than 35%

        # Process each audio segment based on sync mode
        for segment in audio_segments:
            start_ms = int(segment['start_ms'])
            end_ms = int(segment['end_ms'])
            target_duration_ms = end_ms - start_ms
            audio = segment['audio']
            audio_duration_ms = len(audio)

            if time_sync_mode == "stretch":
                # Stretch or compress audio to match target duration with bounds
                if audio_duration_ms != target_duration_ms:
                    speed_ratio = audio_duration_ms / target_duration_ms

                    # Apply speed bounds to prevent unnatural distortion
                    if speed_ratio < MIN_SPEED_RATIO:
                        print(f"   Warning: Segment {segment['index']} requires extreme speedup (ratio={speed_ratio:.2f}), clamping to {MIN_SPEED_RATIO}")
                        speed_ratio = MIN_SPEED_RATIO
                    elif speed_ratio > MAX_SPEED_RATIO:
                        print(f"   Warning: Segment {segment['index']} requires extreme slowdown (ratio={speed_ratio:.2f}), clamping to {MAX_SPEED_RATIO}")
                        speed_ratio = MAX_SPEED_RATIO

                    # Apply speed adjustment using frame rate modification
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * speed_ratio)
                    }).set_frame_rate(audio.frame_rate)

                    actual_duration_ms = len(audio)
                    print(f"   Segment {segment['index']}: adjusted {audio_duration_ms}ms -> {actual_duration_ms}ms (target: {target_duration_ms}ms, ratio: {speed_ratio:.2f})")

                # Insert audio at exact position (non-overlapping)
                # If audio is still longer/shorter due to clamping, adjust placement
                audio_len = len(audio)
                if audio_len <= target_duration_ms:
                    final_audio = final_audio[:start_ms] + audio + final_audio[start_ms + audio_len:]
                else:
                    # If still too long after clamping, crop the end
                    audio = audio[:target_duration_ms]
                    final_audio = final_audio[:start_ms] + audio + final_audio[end_ms:]

            elif time_sync_mode == "crop":
                # Crop audio if it exceeds target duration, with gentle fade
                if audio_duration_ms > target_duration_ms:
                    # Add 50ms fade out to avoid abrupt cuts
                    fade_duration = min(50, target_duration_ms // 4)
                    audio = audio[:target_duration_ms].fade_out(duration=fade_duration)
                    print(f"   Segment {segment['index']}: cropped {audio_duration_ms}ms -> {target_duration_ms}ms (with fade)")
                else:
                    print(f"   Segment {segment['index']}: kept at {audio_duration_ms}ms")

                # Insert audio at exact position
                final_audio = final_audio[:start_ms] + audio + final_audio[start_ms + len(audio):]

            elif time_sync_mode == "overlay":
                # Original overlay behavior (may cause overlapping)
                final_audio = final_audio.overlay(audio, position=start_ms)

            else:
                raise ValueError(f"Invalid time_sync_mode: {time_sync_mode}")

        # Prepare output paths
        output_dir = "/oomol-driver/oomol-storage/srt-synthesis-output"
        os.makedirs(output_dir, exist_ok=True)

        output_audio_path = os.path.join(output_dir, f"srt_synthesis_{int(time.time())}.wav")
        output_srt_path = os.path.join(output_dir, f"srt_metadata_{int(time.time())}.srt")

        # Export final audio
        print(f">> Exporting final audio: {output_audio_path}")
        final_audio.export(output_audio_path, format="wav")

        # Write metadata SRT
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_lines))

        print(f">> SRT synthesis completed successfully!")
        print(f"   Total subtitles: {len(subs)}")
        print(f"   Processed: {len(audio_segments)}")
        print(f"   Audio duration: {len(final_audio) / 1000:.2f}s")

        # Preview audio in OOMOL UI
        context.preview({
            "type": "audio",
            "data": output_audio_path
        })

        return {
            "audio": output_audio_path,
            "srt_with_metadata": output_srt_path
        }

    finally:
        # Clean up temporary directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory: {e}")
