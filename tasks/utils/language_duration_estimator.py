"""
Language-aware duration estimation for TTS synthesis.
Provides better duration prediction for Chinese and English text.
"""

import re
from typing import Tuple, Literal


def detect_language(text: str) -> Literal["zh", "en", "mixed", "other"]:
    """
    Detect the primary language of text.

    Args:
        text: Input text to analyze

    Returns:
        Language code: "zh" (Chinese), "en" (English), "mixed", or "other"
    """
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))

    # Count English words (simple approximation)
    english_words = len(re.findall(r'[a-zA-Z]+', text))

    # Count total characters
    total_chars = len(text.strip())

    if total_chars == 0:
        return "other"

    # Calculate ratios
    chinese_ratio = chinese_chars / total_chars
    english_ratio = (sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))) / total_chars if total_chars > 0 else 0

    # Decision logic
    if chinese_ratio > 0.3 and english_ratio > 0.2:
        return "mixed"
    elif chinese_ratio > 0.3:
        return "zh"
    elif english_ratio > 0.5:
        return "en"
    else:
        return "other"


def estimate_base_duration(text: str, language: str = None) -> float:
    """
    Estimate base speech duration based on text and language.

    Args:
        text: Input text to synthesize
        language: Language code (auto-detected if None)

    Returns:
        Estimated duration in seconds
    """
    if language is None:
        language = detect_language(text)

    # Language-specific speech rate models
    # Based on typical speech rates: Chinese 3-5 chars/sec, English 2.5-3.5 words/sec

    if language == "zh":
        # Chinese: count characters (including punctuation)
        char_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        # Average: 4 characters per second
        base_duration = char_count / 4.0

    elif language == "en":
        # English: count words
        words = re.findall(r'[a-zA-Z]+', text)
        word_count = len(words)
        # Average: 2.8 words per second
        base_duration = word_count / 2.8

    elif language == "mixed":
        # Mixed language: combine both estimations
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))

        base_duration = (chinese_chars / 4.0) + (english_words / 2.8)

    else:
        # Fallback: use character count
        char_count = len(text.strip())
        base_duration = char_count / 10.0

    # Add baseline pause time for punctuation
    punctuation_count = len(re.findall(r'[.,;:!?]', text))
    pause_time = punctuation_count * 0.2  # 200ms per punctuation

    return base_duration + pause_time


def calculate_optimal_mel_tokens(
    text: str,
    target_duration: float,
    language: str = None,
    reference_speed_ratio: float = 1.0
) -> int:
    """
    Calculate optimal max_mel_tokens based on target duration and language.

    Args:
        text: Input text
        target_duration: Target duration in seconds
        language: Language code (auto-detected if None)
        reference_speed_ratio: Speed ratio from reference audio analysis

    Returns:
        Recommended max_mel_tokens value (50-1500)
    """
    if language is None:
        language = detect_language(text)

    # Language-specific mel token rates (tokens per second)
    # These are empirical values based on IndexTTS2 model behavior
    mel_token_rates = {
        "zh": 125,   # Chinese: ~125 tokens/sec
        "en": 100,   # English: ~100 tokens/sec
        "mixed": 115, # Mixed: average
        "other": 110
    }

    base_rate = mel_token_rates.get(language, 110)

    # Adjust for reference audio speed
    adjusted_rate = base_rate * reference_speed_ratio

    # Calculate tokens with 15% safety margin
    estimated_tokens = int(target_duration * adjusted_rate * 1.15)

    # Clamp to valid range
    optimal_tokens = max(50, min(estimated_tokens, 1500))

    return optimal_tokens


def estimate_synthesis_duration(
    text: str,
    max_mel_tokens: int,
    language: str = None,
    reference_speed_ratio: float = 1.0
) -> float:
    """
    Estimate actual synthesis duration given max_mel_tokens.
    Inverse function of calculate_optimal_mel_tokens.

    Args:
        text: Input text
        max_mel_tokens: Max mel tokens parameter
        language: Language code (auto-detected if None)
        reference_speed_ratio: Speed ratio from reference audio

    Returns:
        Estimated synthesis duration in seconds
    """
    if language is None:
        language = detect_language(text)

    mel_token_rates = {
        "zh": 125,
        "en": 100,
        "mixed": 115,
        "other": 110
    }

    base_rate = mel_token_rates.get(language, 110)
    adjusted_rate = base_rate * reference_speed_ratio

    # Account for the 15% safety margin used in calculate_optimal_mel_tokens
    estimated_duration = (max_mel_tokens / 1.15) / adjusted_rate

    return estimated_duration


def get_trim_strategy(language: str) -> dict:
    """
    Get language-specific audio trimming strategy.

    Args:
        language: Language code

    Returns:
        Dictionary with trimming parameters
    """
    strategies = {
        "zh": {
            "silence_threshold_db": -38.0,  # More aggressive silence detection
            "min_silence_len_ms": 80,       # Shorter minimum silence
            "edge_fade_ms": 30,             # Shorter fade
            "leading_bias": 0.4,            # Trim more from start (40%)
            "trailing_bias": 0.6            # Trim more from end (60%)
        },
        "en": {
            "silence_threshold_db": -42.0,  # More conservative
            "min_silence_len_ms": 120,      # Longer minimum silence
            "edge_fade_ms": 50,             # Longer fade for smoother transitions
            "leading_bias": 0.5,            # Equal trimming
            "trailing_bias": 0.5
        },
        "mixed": {
            "silence_threshold_db": -40.0,
            "min_silence_len_ms": 100,
            "edge_fade_ms": 40,
            "leading_bias": 0.45,
            "trailing_bias": 0.55
        },
        "other": {
            "silence_threshold_db": -40.0,
            "min_silence_len_ms": 100,
            "edge_fade_ms": 40,
            "leading_bias": 0.5,
            "trailing_bias": 0.5
        }
    }

    return strategies.get(language, strategies["other"])


def suggest_speed_factor(
    estimated_duration: float,
    target_duration: float,
    language: str,
    max_speed_change: float = 0.25
) -> Tuple[float, str]:
    """
    Suggest optimal speed_factor to match target duration.

    Args:
        estimated_duration: Estimated synthesis duration
        target_duration: Target duration from SRT
        language: Language code
        max_speed_change: Maximum allowed speed change (default 25%)

    Returns:
        Tuple of (speed_factor, recommendation_message)
    """
    if target_duration <= 0 or estimated_duration <= 0:
        return 1.0, "Invalid duration values"

    duration_ratio = estimated_duration / target_duration

    # Calculate ideal speed factor
    ideal_factor = duration_ratio

    # Apply language-specific constraints
    if language == "zh":
        # Chinese can tolerate slightly more speed variation
        max_factor = 1.0 + max_speed_change
        min_factor = 1.0 - max_speed_change
    elif language == "en":
        # English is more sensitive to speed changes
        max_factor = 1.0 + (max_speed_change * 0.8)
        min_factor = 1.0 - (max_speed_change * 0.8)
    else:
        max_factor = 1.0 + max_speed_change
        min_factor = 1.0 - max_speed_change

    # Clamp to safe range
    speed_factor = max(min_factor, min(ideal_factor, max_factor))

    # Generate recommendation message
    if abs(speed_factor - 1.0) < 0.05:
        message = "Duration matches well, no speed adjustment needed"
    elif speed_factor > 1.0:
        message = f"Speech is {((speed_factor - 1) * 100):.1f}% too slow, speeding up"
    else:
        message = f"Speech is {((1 - speed_factor) * 100):.1f}% too fast, slowing down"

    return speed_factor, message
