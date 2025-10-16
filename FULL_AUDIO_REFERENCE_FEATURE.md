# Full Audio Reference Feature

## Overview

New feature that allows SRT synthesis to use **full audio** input and automatically extract time-aligned reference segments for each subtitle. This preserves the original speaker's voice characteristics, prosody, and emotion for each segment.

## Key Benefits

1. **Natural Voice Matching**: Each synthesized segment uses the corresponding time-aligned audio as reference
2. **Prosody Preservation**: Maintains the original rhythm, intonation, and speaking style
3. **Emotion Consistency**: Captures emotional state from the actual spoken audio
4. **Dynamic Reference**: No need to manually select different reference clips

## New Parameters

### `use_full_audio_reference` (boolean)
- **Default**: `false`
- **Description**: Enable full audio reference mode
- **When enabled**: The system will extract audio segments based on SRT timestamps

### `reference_audio_offset_ms` (integer)
- **Default**: `0`
- **Range**: `-5000` to `5000` milliseconds
- **Description**: Time offset adjustment for reference audio
  - **Positive values**: Delay (shift forward)
  - **Negative values**: Advance (shift backward)
- **Use case**: Compensate for audio/subtitle sync issues

## How It Works

### Workflow

```
┌─────────────────┐
│ Full Audio File │
└────────┬────────┘
         │
         ├─► Validate duration matches SRT
         │
         └─► For each subtitle segment:
             ├─► Extract audio at [start_time, end_time]
             ├─► Add padding (100ms) at edges
             ├─► Add fade in/out for smoothness
             ├─► Validate segment quality
             └─► Use as reference for synthesis
```

### Audio Segment Extraction

**Function**: `extract_audio_segment()`

**Features**:
- **Time alignment**: Extracts based on SRT timestamps
- **Smart padding**: Adds 100ms padding at edges for smoother reference
- **Duration constraints**:
  - Minimum: 0.5 seconds (extends or loops if shorter)
  - Maximum: 10 seconds (trims from center if longer)
- **Quality validation**: Checks RMS energy and silence ratio
- **Fade effects**: Smooth fade in/out to avoid clicks

### Quality Validation

**Function**: `analyze_segment_quality()`

Validates extracted segments for:
- **Minimum RMS threshold**: `0.01` (ensures audible content)
- **Maximum silence ratio**: `50%` (ensures sufficient voice content)
- **Fallback**: Uses full audio if segment quality is poor

## Usage Examples

### Example 1: Basic Usage

```yaml
inputs:
  srt_file: subtitles.srt
  spk_audio_prompt: full_audio.wav  # Full audio file
  use_full_audio_reference: true
  reference_audio_offset_ms: 0
```

### Example 2: With Time Offset

If audio and subtitles are slightly out of sync:

```yaml
inputs:
  srt_file: subtitles.srt
  spk_audio_prompt: full_audio.wav
  use_full_audio_reference: true
  reference_audio_offset_ms: -200  # Audio is 200ms ahead, adjust backward
```

## Technical Implementation

### New Utility Module

**File**: `tasks/utils/audio_segment_extractor.py`

**Key Functions**:

1. **`extract_audio_segment()`**
   - Extracts time-aligned audio segments
   - Handles padding, duration constraints, and fading

2. **`validate_full_audio_duration()`**
   - Validates audio duration matches SRT max timestamp
   - Allows 5-second tolerance

3. **`analyze_segment_quality()`**
   - Analyzes RMS energy and silence ratio
   - Returns quality assessment

### Integration Points

**File**: `tasks/srt-synthesis/__init__.py`

**Changes**:
1. Added new input parameters parsing
2. Validates full audio duration before processing
3. Extracts reference segment for each subtitle
4. Falls back to full audio if extraction fails

## Validation Logic

### Duration Validation

```python
# Check if audio duration matches SRT
audio_duration ≈ max_srt_timestamp ± 5 seconds
```

If validation fails, raises error with helpful message.

### Segment Quality Check

```python
# For each extracted segment:
if rms_energy < 0.01 or silence_ratio > 50%:
    fallback_to_full_audio()
else:
    use_extracted_segment()
```

## Comparison: Static vs. Full Audio Reference

| Aspect | Static Reference | Full Audio Reference |
|--------|------------------|----------------------|
| Input | Single audio clip | Full audio file |
| Voice consistency | Same for all segments | Time-aligned per segment |
| Prosody | Static reference prosody | Original prosody preserved |
| Emotion | Static emotion | Dynamic emotion per segment |
| Setup | Manual clip selection | Automatic extraction |
| Quality | Uniform | Matches original timing |

## Best Practices

### 1. Audio Quality
- Use **lossless formats** (WAV, FLAC) for best quality
- Ensure **clean audio** without background noise
- Match **sample rate** with TTS model requirements

### 2. Time Sync
- Verify **SRT timestamps** match audio timing
- Use **offset parameter** if sync is off
- Test with a few segments first

### 3. Segment Duration
- Ideal segment length: **1-5 seconds**
- Avoid very short segments (< 0.5s)
- System handles long segments automatically

### 4. Fallback Strategy
- System uses **full audio** if segment extraction fails
- Check logs for quality warnings
- Manual intervention may be needed for poor quality segments

## Limitations

1. **Audio/SRT Sync Required**: Audio and subtitles must be synchronized
2. **Quality Dependent**: Extracted segments must have sufficient voice content
3. **Duration Constraints**: Very short or very long segments may be adjusted
4. **Single Speaker**: Works best with single-speaker audio

## Troubleshooting

### Issue: "Duration validation failed"
**Solution**: Check that audio file covers the entire SRT duration. Adjust tolerance or fix sync.

### Issue: "Segment quality too low"
**Solution**: Check if the time range contains clear speech. May need to adjust SRT timestamps.

### Issue: "Audio too quiet"
**Solution**: Normalize audio levels before using as reference.

### Issue: "Too much silence"
**Solution**: Check if SRT timestamps are accurate. May need manual correction.

## Performance Considerations

- **Extraction overhead**: ~50-100ms per segment
- **Memory usage**: Loads full audio once, extracts segments incrementally
- **Disk space**: Temporary segment files cleaned up after synthesis
- **Total time**: Adds ~5-10% to synthesis time

## Future Enhancements

Potential improvements:
1. **Auto-sync detection**: Automatically detect and correct audio/SRT offset
2. **Multi-speaker support**: Detect different speakers and use appropriate segments
3. **Quality enhancement**: Pre-process segments to improve quality
4. **Caching**: Cache extracted segments for repeated use
5. **Adaptive extraction**: Adjust extraction strategy based on segment characteristics

## Example Use Cases

### 1. Video Dubbing
- Extract audio from original video
- Generate SRT with translated text
- Synthesize with full audio reference
- Maintain original speaking style

### 2. Podcast Re-recording
- Use original podcast audio
- Edit transcript in SRT
- Re-synthesize with time-aligned voice
- Preserve natural flow

### 3. Audiobook Production
- Use narrator's sample recording
- Generate from full script
- Match narrator's cadence
- Consistent voice throughout

## Migration Guide

### From Static Reference

**Before**:
```yaml
spk_audio_prompt: short_clip.wav  # 3-second clip
use_full_audio_reference: false
```

**After**:
```yaml
spk_audio_prompt: full_audio.wav  # Full-length audio
use_full_audio_reference: true
reference_audio_offset_ms: 0
```

### Compatibility

- **Backward compatible**: Default `use_full_audio_reference=false` preserves old behavior
- **No breaking changes**: Existing workflows continue to work
- **Gradual adoption**: Can test on single flows before full migration
