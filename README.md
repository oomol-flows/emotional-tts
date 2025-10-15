# IndexTTS2 Text-to-Speech for OOMOL

Transform any text into natural, emotionally expressive speech with AI-powered voice cloning.

## What is This?

This OOMOL package provides a powerful text-to-speech system that can:

- **Clone any voice** from a short audio sample (just 3-15 seconds!)
- **Control emotions** in the generated speech (happy, sad, angry, calm, etc.)
- **Support multiple languages** including Chinese and English
- **Generate high-quality audio** that sounds natural and expressive

Perfect for content creators, audiobook producers, accessibility applications, or anyone who needs realistic AI voices.

## Available Blocks

### IndexTTS2 Speech Synthesis

**What it does:** Converts your text into spoken audio using an AI voice that you provide.

**Key Features:**

- **Voice Cloning:** Upload a short recording of any voice, and the AI will mimic that voice
- **Emotion Control:** Choose how the voice should sound (excited, calm, worried, etc.)
- **Multi-language:** Works with both Chinese and English text
- **Professional Quality:** Produces studio-grade audio suitable for production use

### SRT Subtitle Synthesis

**What it does:** Converts SRT subtitle files into synchronized spoken audio, automatically aligning generated speech with subtitle timestamps.

**Key Features:**

- **Automatic Timing:** Generates audio synchronized with your subtitle timestamps
- **Batch Processing:** Processes all subtitle entries in one go
- **Full Emotion Control:** Uses the same emotion control features as the basic synthesis block
- **Speed Adjustment:** Optionally adjust speech speed to better fit subtitle timing
- **Metadata Output:** Generates an annotated SRT file with audio duration information

### Video Speech Translation (Subflow)

**What it does:** Translates video speech from one language to another while preserving the original speaker's voice characteristics. This is a complete end-to-end pipeline that automatically processes your video.

**Key Features:**

- **Complete Pipeline:** Extract audio → Transcribe → Translate → Synthesize → Replace audio
- **Voice Preservation:** Maintains the original speaker's voice timbre in the target language
- **Multi-language Support:** Supports all major languages (English, Chinese, Spanish, French, German, Japanese, Korean, Russian, Arabic, Portuguese, Italian, Dutch, Polish, Turkish, Vietnamese, Thai, Indonesian, Hindi)
- **Auto-sync:** Automatically synchronizes translated speech with original video timing
- **One-click Operation:** Upload your video, select languages, and get a translated version

## Use Cases

### 1. Content Creation
- Generate voiceovers for videos without recording
- Create podcast episodes in different voices
- Produce audiobooks with consistent narration
- **NEW:** Convert video subtitles to synchronized narration tracks

### 2. Accessibility
- Convert written content to audio for visually impaired users
- Create audio versions of documents and articles
- Generate multilingual audio content
- **NEW:** Generate audio descriptions from subtitle files

### 3. Prototyping & Design
- Test different voice styles for your project
- Create audio mockups without hiring voice actors
- Experiment with emotional tones for storytelling

### 4. Education & Training
- Produce educational audio materials
- Create language learning content
- Generate consistent instructional narration
- **NEW:** Create synchronized audio lessons from subtitle transcripts

### 5. Video Production
- **NEW:** Add narration to silent videos using subtitle files
- **NEW:** Create dubbed versions with customized voices
- **NEW:** Generate audio tracks synchronized with existing subtitles

### 6. Video Localization & Translation
- **NEW:** Translate video content to different languages while keeping the original voice
- **NEW:** Create multilingual versions of educational videos or tutorials
- **NEW:** Localize marketing videos for international audiences
- **NEW:** Generate dubbed versions of interviews or documentaries

## How to Use

### Basic Text-to-Speech Usage

1. **Prepare Your Inputs:**
   - **Text:** The content you want to convert to speech
   - **Voice Sample:** A 3-15 second audio file of the voice you want to clone (WAV, MP3, or FLAC)

2. **Add the Block to Your Flow:**
   - Drag the "IndexTTS2 Speech Synthesis" block into your OOMOL workflow
   - Connect your text input
   - Upload your voice reference audio

3. **Choose Emotion Control Mode:**
   - **Speaker Mode** (Default): Use the emotion from your voice sample
   - **Reference Mode**: Use a different audio file to control emotion
   - **Vector Mode**: Use sliders to mix different emotions
   - **Text Mode**: Describe the emotion you want in words

4. **Run Your Flow:**
   - The block will generate a WAV audio file with your synthesized speech

### SRT Subtitle-to-Speech Usage

1. **Prepare Your Inputs:**
   - **SRT File:** Your subtitle file (must be in valid SRT format with timestamps)
   - **Voice Sample:** A 3-15 second audio file of the voice you want to clone

2. **Add the Block to Your Flow:**
   - Drag the "SRT Subtitle Synthesis" block into your OOMOL workflow
   - Upload your SRT subtitle file
   - Upload your voice reference audio
   - (Optional) Choose emotion control mode and adjust settings

3. **Configure Optional Settings:**
   - **Speed Factor:** Adjust speech speed (0.5-2.0x, default: 1.0)
   - **Emotion Controls:** Same options as basic synthesis
   - **Quality Parameters:** Temperature, top-p, top-k settings

4. **Run Your Flow:**
   - The block will process all subtitles and generate:
     - A synchronized audio file (WAV) aligned with subtitle timestamps
     - A metadata SRT file showing audio durations vs. target durations

**How It Works:**
- Each subtitle entry is synthesized individually
- Generated audio segments are placed at their correct timestamps
- Silence is automatically added between segments
- The final audio duration matches the last subtitle's end time

### Video Speech Translation Usage

1. **Prepare Your Input:**
   - **Video File:** Any video file (MP4, AVI, MOV, MKV, WebM, FLV)
   - **Language Settings:**
     - Source language (or select "Auto Detect")
     - Target language for translation

2. **Add the Subflow to Your Workflow:**
   - Drag the "Video Speech Translation" subflow into your OOMOL workflow
   - Upload your video file
   - Select source and target languages

3. **How It Works Behind the Scenes:**
   - **Step 1:** Extracts audio from the video
   - **Step 2:** Transcribes the audio to SRT subtitles
   - **Step 3:** Translates subtitles to target language
   - **Step 4:** Generates a voice sample from the original audio (first 25 seconds)
   - **Step 5:** Synthesizes translated speech using the cloned voice
   - **Step 6:** Replaces the original audio track with translated speech

4. **Run Your Flow:**
   - The subflow will automatically execute all steps
   - Output: A video file with translated speech in the target language
   - The original speaker's voice characteristics are preserved

**Use Case Example:** You have an English tutorial video and want to create a Chinese version. The subflow will automatically extract audio, transcribe it, translate to Chinese, clone the speaker's voice, generate Chinese speech that sounds like the original speaker, and replace the audio track - all in one automated pipeline!

### Emotion Control Options

#### Simple: Speaker Mode
The generated speech matches the emotion in your voice sample. If your sample sounds happy, the output will sound happy.

#### Advanced: Reference Mode
Use two audio files:
- **Voice Sample:** Provides the voice timbre (who it sounds like)
- **Emotion Sample:** Provides the emotional style (how it sounds)

Example: Clone Person A's voice but use Person B's energetic speaking style.

#### Precise: Vector Mode
Fine-tune emotion using 8 sliders:
- Happy
- Angry
- Sad
- Afraid
- Disgusted
- Melancholic
- Surprised
- Calm

Mix and match intensities (0.0 to 1.0) to create the perfect emotional tone.

#### Creative: Text Mode
Describe the emotion you want in natural language:
- "The danger is approaching quietly"
- "You scared me to death! Are you a ghost?"
- "This is the best news I've heard all day"

The AI will interpret your description and apply that emotional style.

## Requirements

### System Requirements
- **GPU:** CUDA-capable GPU recommended (NVIDIA)
- **Memory:** 12GB VRAM (for GPU) or 16GB RAM (for CPU)
- **Storage:** ~20GB free space for models and dependencies

### Audio Requirements
- **Voice Sample Quality:** Clear, noise-free recording
- **Duration:** 3-15 seconds (longer is not always better)
- **Single Speaker:** Only one person speaking
- **Format:** WAV, MP3, FLAC, or M4A

## Installation

### Automatic Setup
This package is fully self-contained. On first use, OOMOL will automatically:

1. Install all required Python packages
2. Download the IndexTTS2 AI model (~10GB)
3. Set up the processing environment

**First-time setup takes 10-20 minutes** depending on your internet speed. After that, the block is ready to use instantly.

### Manual Setup (Optional)
If automatic setup fails, the models are stored in:
```
/oomol-driver/oomol-storage/indextts-checkpoints/
```

## Tips for Best Results

### Voice Quality
✅ **Do:**
- Use clean, professional-quality audio
- Record in a quiet environment
- Use a single speaker only
- Provide 5-10 seconds of clear speech

❌ **Avoid:**
- Noisy or echo-filled recordings
- Multiple speakers talking
- Music or background sounds
- Very short samples (<3 seconds)

### SRT Subtitle Files
✅ **Best Practices:**
- Use properly formatted SRT files with valid timestamps
- Keep subtitle text concise (shorter segments work better)
- Ensure subtitle durations are reasonable (at least 2-3 seconds per entry)
- Use UTF-8 encoding for international characters
- Test with speed_factor adjustment if audio is too fast/slow for timing

⚠️ **Known Limitations:**
- Very short subtitle durations (<1 second) may not fit the generated audio
- Very long subtitle texts may be truncated (use max_mel_tokens parameter)
- The generated audio is placed at exact timestamps (may overlap if subtitles are too close)

### Emotion Control
- **For natural results:** Use Speaker mode with a well-recorded sample
- **For creative control:** Try Text mode with emotion weight 0.6-0.7
- **For precise tuning:** Use Vector mode, keeping total emotion sum ≤ 0.8
- **For dramatic effect:** Combine Reference mode with expressive emotion samples

### Text Input
- Longer texts are automatically split into sentences
- Supports punctuation for natural pauses
- Works best with properly formatted text
- Both English and Chinese are supported

## Advanced Parameters

For users who want more control:

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Temperature** | 0.1 - 2.0 | Higher = more varied, Lower = more stable (default: 0.8) |
| **Emotion Weight** | 0.0 - 1.0 | How strong the emotion should be (default: 0.65) |
| **Max Audio Length** | 50 - 1500 tokens | Maximum length of generated audio |
| **Sentence Splitting** | 20 - 200 tokens | How to split long texts into segments |

## Troubleshooting

### Audio Quality Issues
- **Problem:** Voice doesn't sound like the reference
  - **Solution:** Use a higher-quality reference audio (clear, noise-free)

- **Problem:** Emotion is too strong or too weak
  - **Solution:** Adjust the emotion weight (try 0.5-0.8 range)

- **Problem:** Voice sounds robotic
  - **Solution:** Lower the temperature to 0.6-0.7 for more stability

### Technical Issues
- **Problem:** Out of memory error
  - **Solution:** Reduce max audio length or use shorter text inputs

- **Problem:** Model not found
  - **Solution:** Wait for automatic download, or restart OOMOL to retry

- **Problem:** Slow processing
  - **Solution:** GPU processing is 10-20x faster than CPU

## Model Information

- **Model:** IndexTTS2 by Bilibili AI
- **Technology:** Zero-shot text-to-speech with emotion control
- **Languages:** Chinese, English
- **License:** Bilibili IndexTTS License
- **Source:** [github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)
- **Research Paper:** [arXiv:2506.21619](https://arxiv.org/abs/2506.21619)

### Commercial Use
For commercial licensing inquiries, contact: indexspeech@bilibili.com

## Support & Community

- **Demo:** Try the online demo at [HuggingFace Space](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)
- **Issues:** Report bugs on [GitHub](https://github.com/index-tts/index-tts/issues)
- **Discord:** Join the community at https://discord.gg/uT32E7KDmy
- **QQ Groups:** 553460296, 663272642 (Chinese community)

## Example Workflows

### Basic Text-to-Speech Workflow

1. **Text Input Block** → Type your script
2. **File Input Block** → Upload your voice sample (5-10 seconds)
3. **IndexTTS2 Block** → Connect text and audio inputs
4. **Audio Output Block** → Save or play the generated speech

### SRT Subtitle Synthesis Workflow

1. **File Input Block** → Upload your SRT subtitle file
2. **File Input Block** → Upload your voice sample (5-10 seconds)
3. **SRT Subtitle Synthesis Block** → Connect both file inputs
4. **Audio Output Block** → Save the synchronized audio track
5. **File Output Block** → Save the metadata SRT file (optional)

**Use Case Example:** You have a video with subtitles but no audio. Upload the SRT file, provide a voice sample, and generate a synchronized narration track that matches the subtitle timing perfectly!

### Video Speech Translation Workflow

1. **File Input Block** → Upload your source video file (e.g., English.mp4)
2. **Video Speech Translation Subflow** → Connect the video input
   - Set source language to "Auto Detect" or specific language
   - Set target language (e.g., Chinese, Spanish, etc.)
3. **File Output Block** → Save the translated video

**Use Case Example:** You have a 10-minute English tutorial video and want to create versions in Chinese, Spanish, and French. Use this subflow three times with different target languages, and you'll get three localized versions with the same speaker's voice speaking each language naturally!

## License

This package uses IndexTTS2, which is licensed under the Bilibili IndexTTS License. Please review the license terms before commercial use.

## Credits

- **IndexTTS2 Model:** Bilibili AI Team
- **Research:** Zhou et al. (2025)
- **OOMOL Package:** alwaysmavs

---

**Ready to start?** Add the IndexTTS2 block to your OOMOL workflow and bring your text to life with AI voices!
