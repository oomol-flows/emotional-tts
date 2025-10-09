# IndexTTS2 OOMOL Task Block

IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech

## Overview

This OOMOL task block provides an interface to IndexTTS2, a state-of-the-art zero-shot text-to-speech system with:

- **Voice Cloning**: Clone any voice from a short audio reference (3-15 seconds)
- **Emotion Control**: Four different emotion control modes
- **Multi-language Support**: Chinese and English synthesis
- **High Quality**: Emotionally expressive and natural-sounding speech

## Features

### Emotion Control Modes

1. **Speaker Mode** - Use the emotion from the speaker's voice reference
2. **Reference Mode** - Use a separate emotional reference audio
3. **Vector Mode** - Control 8 emotion dimensions via sliders:
   - Happy, Angry, Sad, Afraid
   - Disgusted, Melancholic, Surprised, Calm
4. **Text Mode** - Describe emotions in natural language (e.g., "危险在悄悄逼近")

### Advanced Parameters

- **Temperature** (0.1-2.0): Controls diversity vs stability
- **Top-p/Top-k**: Nucleus and top-k sampling parameters
- **Max Mel Tokens**: Maximum audio length
- **Text Segmentation**: Automatic sentence splitting for long texts

## Installation

The task block is fully self-contained and automatically sets up everything on first use:

1. **IndexTTS Package**: Installed from GitHub (`pip install git+https://github.com/index-tts/index-tts.git`)
2. **Model Files**: Downloaded from HuggingFace (~10GB)
3. **Dependencies**: All Python packages installed via Poetry

### Requirements

- Python 3.10-3.12
- CUDA-capable GPU (recommended)
- ~12GB VRAM for FP16 inference
- ~20GB disk space for models and packages

### Bootstrap Process

The bootstrap script automatically:
1. Installs Node.js dependencies
2. Installs Python dependencies via Poetry
3. Installs IndexTTS package from GitHub
4. Downloads IndexTTS2 model checkpoints to `/oomol-driver/oomol-storage/indextts-checkpoints/`

All setup is handled automatically - no manual intervention required!

## Usage

### Basic Example

1. **Input**:
   - Text: "欢迎大家来体验IndexTTS2"
   - Speaker Audio: Upload a 3-15 second voice sample
   - Emotion Mode: Speaker

2. **Output**: Generated speech audio file (.wav)

### Advanced Usage

#### Emotion Reference Mode
Use a separate audio file to control the emotion while keeping the speaker's voice:
- Speaker Audio: Person A's voice (timbre)
- Emotion Audio: Person B's emotional speech (emotion style)
- Result: Person A's voice with Person B's emotion

#### Emotion Vector Mode
Fine-tune emotion intensities:
- Happy: 0.3
- Surprised: 0.45
- Calm: 0.0
- Others: 0.0

#### Text Emotion Mode
Describe the emotion in text:
- Text: "快躲起来!是他要来了!"
- Emotion Text: "你吓死我了!你是鬼吗?"
- Emotion Weight: 0.6

## Parameters Reference

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| text | string | - | Required | Text to synthesize |
| spk_audio_prompt | file | .wav/.mp3 | Required | Voice reference audio |
| emo_control_mode | enum | 4 modes | speaker | Emotion control method |
| emo_audio_prompt | file | .wav/.mp3 | null | Emotional reference audio |
| emo_weight | number | 0.0-1.0 | 0.65 | Emotion intensity |
| emo_text | string | - | null | Emotion description |
| emo_vec_* | number | 0.0-1.0 | 0.0 | 8 emotion sliders |
| use_random | boolean | - | false | Random sampling |
| temperature | number | 0.1-2.0 | 0.8 | Sampling temperature |
| top_p | number | 0.0-1.0 | 0.8 | Nucleus sampling |
| top_k | integer | 0-100 | 30 | Top-k sampling |
| max_mel_tokens | integer | 50-1500 | 1500 | Max audio tokens |
| max_text_tokens_per_segment | integer | 20-200 | 120 | Sentence splitting |

## Model Information

- **Model**: IndexTTS2 (Bilibili AI)
- **Version**: 2.0.0
- **License**: Bilibili IndexTTS License
- **Source**: [github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)
- **Paper**: [arXiv:2506.21619](https://arxiv.org/abs/2506.21619)

## Tips

1. **Voice Reference Quality**:
   - Use clear, noise-free audio (3-15 seconds)
   - Single speaker only
   - Higher quality = better cloning

2. **Emotion Control**:
   - For text mode, use emo_weight 0.6-0.7
   - Vector mode: total emotion sum should be ≤ 0.8
   - Reference mode works best with clear emotional expression

3. **Performance**:
   - FP16 inference enabled by default
   - RTF (Real-Time Factor) typically 0.1-0.3x
   - Longer texts auto-split into segments

4. **Text Formatting**:
   - Supports Pinyin control: "DE5" for tone 5
   - See `/oomol-driver/oomol-storage/indextts-checkpoints/pinyin.vocab`

## Troubleshooting

### Out of Memory
- Reduce `max_mel_tokens`
- Reduce `max_text_tokens_per_segment`
- Use shorter text inputs

### Audio Quality Issues
- Check reference audio quality
- Adjust `temperature` (lower = more stable)
- Try different `emo_weight` values

### Model Not Found
- Bootstrap script will auto-download on first run
- Manual download: `huggingface-cli download IndexTeam/IndexTTS-2 --local-dir /oomol-driver/oomol-storage/indextts-checkpoints`

## Citation

```bibtex
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```

## License

This task block uses IndexTTS2 which is licensed under the Bilibili IndexTTS License. See the LICENSE file for details.

For commercial usage, please contact: indexspeech@bilibili.com

## Support

- GitHub Issues: [index-tts/index-tts](https://github.com/index-tts/index-tts/issues)
- Demo: [HuggingFace Space](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)
- QQ Group: 553460296, 663272642
- Discord: https://discord.gg/uT32E7KDmy