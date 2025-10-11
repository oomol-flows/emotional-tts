#region generated meta
import typing
class Inputs(typing.TypedDict):
    text: str
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
class Outputs(typing.TypedDict):
    audio: typing.NotRequired[str]
#endregion

import os
import sys
import time
from oocana import Context

# Suppress warnings before importing
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import model downloader utility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_downloader import ensure_model_downloaded

# Set HuggingFace cache directories to use larger filesystem
hf_cache_dir = '/oomol-driver/oomol-storage/indextts-checkpoints/hf_cache'
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ['HF_HUB_CACHE'] = hf_cache_dir
os.environ['HF_HOME'] = hf_cache_dir
os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
os.environ['HF_DATASETS_CACHE'] = hf_cache_dir
# Disable XET download acceleration to avoid temp file issues
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

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

def main(params: Inputs, context: Context) -> Outputs:
    """
    IndexTTS2 text-to-speech synthesis task

    Args:
        params: Input parameters including text, voice reference, and emotion controls
        context: OOMOL context object

    Returns:
        Output dictionary with generated audio file path
    """
    # Model directory and config path
    model_dir = "/oomol-driver/oomol-storage/indextts-checkpoints"
    cfg_path = os.path.join(model_dir, "config.yaml")

    # Get TTS model (will download if needed)
    tts = get_tts_model(model_dir, cfg_path)

    # Extract parameters with defaults for nullable fields
    text = params["text"]
    spk_audio_prompt = params["spk_audio_prompt"]
    emo_control_mode = params.get("emo_control_mode") or "speaker"
    emo_audio_prompt = params.get("emo_audio_prompt")
    emo_weight = params.get("emo_weight") if params.get("emo_weight") is not None else 0.65
    emo_text = params.get("emo_text")
    use_random = params.get("use_random") if params.get("use_random") is not None else False
    max_mel_tokens = params.get("max_mel_tokens") if params.get("max_mel_tokens") is not None else 1500
    temperature = params.get("temperature") if params.get("temperature") is not None else 0.8
    top_p = params.get("top_p") if params.get("top_p") is not None else 0.8
    top_k = params.get("top_k") if params.get("top_k") is not None else 30
    max_text_tokens_per_segment = params.get("max_text_tokens_per_segment") if params.get("max_text_tokens_per_segment") is not None else 120

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

    # Validate text input
    if not text or text.strip() == "":
        raise ValueError("Text input cannot be empty")

    # Validate speaker audio prompt
    if not spk_audio_prompt or not os.path.exists(spk_audio_prompt):
        raise ValueError(f"Speaker audio prompt file not found: {spk_audio_prompt}")

    # Prepare output path
    output_dir = "/oomol-driver/oomol-storage/indextts-output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"indextts_{int(time.time())}.wav")

    # Prepare inference parameters based on emotion control mode
    use_emo_text = False
    final_emo_audio = None
    final_emo_vector = None

    if emo_control_mode == "speaker":
        # Mode 0: Use speaker's voice emotion
        final_emo_audio = None  # Will default to spk_audio_prompt
        final_emo_vector = None

    elif emo_control_mode == "reference":
        # Mode 1: Use reference audio emotion
        if emo_audio_prompt and os.path.exists(emo_audio_prompt):
            final_emo_audio = emo_audio_prompt
        else:
            raise ValueError("Emotion reference audio is required for 'reference' mode")
        final_emo_vector = None

    elif emo_control_mode == "vector":
        # Mode 2: Use emotion vector controls
        final_emo_audio = None
        final_emo_vector = emo_vector

    elif emo_control_mode == "text":
        # Mode 3: Use text description emotion
        use_emo_text = True
        final_emo_audio = None
        final_emo_vector = None
        # emo_text can be None (will use main text) or custom description

    else:
        raise ValueError(f"Invalid emotion control mode: {emo_control_mode}")

    print(f">> Synthesizing speech:")
    print(f"   Text: {text[:50]}..." if len(text) > 50 else f"   Text: {text}")
    print(f"   Emotion mode: {emo_control_mode}")
    print(f"   Speaker audio: {spk_audio_prompt}")
    print(f"   Emotion weight: {emo_weight}")

    # Run inference
    try:
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
            verbose=True,
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

        if result_path and os.path.exists(result_path):
            print(f">> Audio generated successfully: {result_path}")

            # Preview audio in OOMOL UI
            context.preview({
                "type": "audio",
                "data": result_path
            })

            return {"audio": result_path}
        else:
            raise ValueError("Audio generation failed: output file not created")

    except Exception as e:
        raise ValueError(f"IndexTTS2 inference failed: {str(e)}")
