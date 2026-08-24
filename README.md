***

> ## 🇮🇷 Persian / فارسی
>
> This fork adds **Persian (Farsi)** support, which upstream Chatterbox does not
> ship among its 23 languages, plus a mechanism for staying on the latest stable
> upstream release. See **[README_FA.md](README_FA.md)** for the full guide.
>
> Quick differences from the instructions below: the Persian path starts from the
> **multilingual** checkpoint rather than the English one, adds a `[fa]` language
> token seeded from `[ar]`, and normalises text through `src/persian/`. Weights
> come from `python tools/fetch_models.py` (not `setup.py`), the corpus from
> `python tools/build_dataset.py`, and generation from `infer_fa.py`.

---

# Chatterbox: Fine-Tuning & LoRA Inference Kit 🎙️

A modular, highly efficient infrastructure for **fine-tuning** both **Chatterbox TTS (Standart)** and **Chatterbox Turbo** models with your own dataset and generating high-quality speech synthesis.

This kit is specially designed to support **new languages** and voices by intelligently extending the model's vocabulary. With the newly added **LoRA (Low-Rank Adaptation)** support, you can now train high-quality voices faster and with significantly less VRAM.

> ### 💡 Developer's Recommendation: Training the Turbo Model
> You can train both the **Standart** and **Turbo** models using either *Full Fine-Tuning* or *LoRA*. 
> 
> If you are aiming to fine-tune the **Turbo model**, you may want to try a **Full Fine-Tune** first. However, the Turbo architecture's weights can sometimes be quite stubborn to adapt. If you start hearing static noise, hallucinations, or meaningless sounds during training (which you can easily monitor via the automatic `inference_callback` audio samples), **we strongly recommend switching to LoRA (`is_lora = True`)**.
> 
> *I have personally trained the Turbo model using LoRA and achieved highly successful, stable results! The LoRA matrices act as a great stabilizer for the base model.*
>
> **The Ultimate Workflow:** Train your model with LoRA, test the adapter directly using `inference.py`, and once you are 100% satisfied with the voice quality, run `python merge_lora.py`. This will bake the LoRA weights into the base model and give you a single, standalone `.safetensors` file ready for production!

---

## 🧠 Training Strategies: LoRA vs. Full Fine-Tune

This repository allows you to choose your training strategy using the `is_lora` flag in `src/config.py`. 

### 1. LoRA Mode (`is_lora = True`) - 🌟 HIGHLY RECOMMENDED
*   **What is it?** LoRA (Low-Rank Adaptation) freezes the massive base model and only trains tiny adapter layers alongside the new language embeddings.
*   **Best for:** Datasets that are **10 hours or less** in total duration (which covers 95% of custom voice cloning use cases).
*   **Benefits:** Prevents the model from forgetting its base knowledge (catastrophic forgetting), trains significantly faster, prevents overfitting on small datasets, and uses **~60% less VRAM**.
*   **Output:** Saves a lightweight adapter to the `new_lang_adapter` folder instead of a multi-gigabyte model file.

### 2. Full Fine-Tune (`is_lora = False`)
*   **What is it?** Unfreezes and updates every single weight inside the T3 Transformer model.
*   **Best for:** Massive, studio-grade datasets that are **strictly larger than 10 hours** where you want to completely overwrite the model's fundamental acoustic understanding.
*   **Drawbacks:** Requires massive GPU VRAM, takes much longer to train, and risks ruining the base model's voice quality if the dataset is too small.

---

## ⚠️ Understanding the Two Modes: Standart vs. Turbo

This repository operates in two distinct modes, controlled by the `is_turbo` setting in `src/config.py`. Please decide which mode you need before you begin.

### 1. Standart Mode (`is_turbo = False`)
*   **Architecture:** Llama-based.
*   **Tokenizer:** **Grapheme (character) based.** The `tokenizer.json` downloaded by `setup.py` contains a small, efficient vocabulary (~2,454 tokens) covering 23 languages.
*   **Best for:** Training a model with full control over a specific language from a more fundamental level.

### 2. Turbo Mode (`is_turbo = True`)
*   **Architecture:** GPT-2 based.
*   **Tokenizer:** **BPE-based.** It starts with a large, powerful English vocabulary (~50,000+ tokens).
*   **Smart Merging:** When you run `setup.py`, this large vocabulary is **automatically extended** with our multi-language grapheme set.
*   **Best for:** Leveraging a strong English base for faster, high-quality fine-tuning on other languages.

---

## ⚠️ **CRITICAL: Switching Between Training Modes**

If you plan to switch between **Standard Mode** (`is_turbo = False`) and **Turbo Mode** (`is_turbo = True`), you **MUST** completely delete the `pretrained_models` directory and the preprocessed_dir directory created with the `preprocess = True` operation before running the `setup.py` file again.

The setup script replaces the token files in place. If you run the setup for Standard mode after setting up for Turbo (or vice versa), the token files will become corrupted and cause errors that are difficult to debug during training.

**Correct Workflow for Changing Modes:**

1. **DELETE the entire `pretrained_models` folder.**

```bash
# On Linux or macOS
rm -rf pretrained_models

# On Windows (in Command Prompt)
rmdir /s /q pretrained_models
```
2. **Update** the `src/config.py` file, setting the `is_turbo` flag to your desired new mode and setting `preprocess = True` if it is False.

3. **RUN setup.py again** to download and prepare the correct files for the new mode.

```bash
python setup.py
```
4. **Update** the `new_vocab_size` value in the `src/config.py` file with the new value provided by the setup script. Also ensure `preprocess = True`.

---

## ⚠️ CRITICAL INFORMATION (Please Read)

### 0. Preprocessing is Mandatory
This repository uses an **offline preprocessing** strategy to maximize training speed. This script processes all audio files, extracts speaker embeddings and acoustic tokens, and saves them as `.pt` files.

### 1. Tokenizer and Vocab Size (Most Important)
Chatterbox uses a grapheme-based (character-level) tokenizer. The `tokenizer.json` file downloaded by `setup.py` includes support for **23 languages** from the original Chatterbox repository, covering most common characters across multiple languages.

*   **Default Support:** The provided tokenizer already includes characters for English, Turkish, French, German, Spanish, and 18+ other languages
*   **When to customize:** If your target language has special characters not covered in the default tokenizer, you can create a custom `tokenizer.json`
*   **Examples of special characters by language:**
    *   Turkish: `ç, ğ, ş, ö, ü, ı`
    *   French: `é, è, ê, à, ù, ç`
    *   German: `ä, ö, ü, ß`
    *   Spanish: `ñ, á, é, í, ó, ú`
*   **Critical:** The `new_vocab_size` variable in `src/config.py` **must exactly match** the total number of tokens in your `tokenizer.json` file
*   **Default vocab size:** Check the downloaded `tokenizer.json` to see the exact token count, then set `new_vocab_size` accordingly.

### 2. Audio Sample Rates
*   **Training (Input):** Chatterbox's encoder and T3 module work with **16,000 Hz (16kHz)** audio. Even if your dataset uses different rates, `dataset.py` automatically resamples to 16kHz.
*   **Output (Inference):** The model's vocoder generates audio at **24,000 Hz (24kHz)**.

---

## 📂 Folder Structure

```text
chatterbox-finetune/
├── pretrained_models/                             # setup.py downloads required models here
├── MyTTSDataset/                                  # Your custom dataset in LJSpeech format
├── FileBasedDataset/                              # Your custom dataset in File-Based format
├── speaker_reference/                             # Speaker reference audio files
├── src/
│   ├── config.py                                  # All settings and hyperparameters
│   ├── dataset.py                                 # Data loading and processing
│   ├── model.py                                   # Model weight transfer and training wrapper
│   ├── inference_callback.py                      # Callbacks for checking audio during training
│   ├── preprocess_*.py                            # Preprocessing scripts (LJSpeech, JSON, etc.)
│   └── utils.py                                   # Logger and VAD utilities
├── train.py                                       # Main training script (Handles LoRA & Full)
├── inference.py                                   # Smart speech synthesis script
├── merge_lora.py                                  # Bakes LoRA weights into the base model
├── setup.py                                       # Setup script for downloading models
├── requirements.txt                               # Required dependencies
└── README.md                                      # This file
```

---

## 🚀 Installation

### 1. Install Dependencies
Requires Python 3.8+ and GPU (recommended):

**Install FFmpeg (Required):**
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

**Install Python Dependencies:**
```bash
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
cd chatterbox-finetuning
pip install -r requirements.txt
```

### 2. Download & Prepare Models (CRITICAL)
This multi-step process prepares all necessary files based on your chosen mode. This script downloads the necessary base models (`ve`, `s3gen`, `t3`) and default tokenizer. **Must be run before training.**

**Step 2.1: Choose Your Mode**
Open `src/config.py` and set the `is_turbo` variable to `True` or `False`.

**Step 2.2: Run the Setup Script**
This command will download the correct model files. If Turbo mode is enabled, it will also **automatically merge the tokenizers for you.**
```bash
python setup.py
```

**Step 2.3: Update Config (Turbo Mode ONLY)**
If you ran the setup in Turbo mode, the script will output a final message like this:
`Please update the 'new_vocab_size' in 'src/config.py' to the following value: 52260`
Copy this exact number and paste it into the `new_vocab_size` variable in `src/config.py`. **Do not skip this step!**

---

## 🏋️ Training (Fine-Tuning)

During training, the script loads the original model weights, **intelligently resizes them** for the new vocabulary size, and initializes new tokens using mean initialization from existing tokens for faster adaptation.

### 1. Dataset Preparation

#### Option A: Using TTS Dataset Generator (Recommended)
We recommend using the [TTS Dataset Generator](https://github.com/gokhaneraslan/tts-dataset-generator) tool to automatically create high-quality datasets from audio or video files.

**Quick Start:**
```bash
# Install the dataset generator
git clone https://github.com/gokhaneraslan/tts-dataset-generator.git
cd tts-dataset-generator
pip install -r requirements.txt

# Generate dataset from your audio/video file
python main.py --file your_audio.mp4 --model large --language en --ljspeech True
```
This will automatically segment audio, transcribe it via Whisper AI, and format it for the `MyTTSDataset/` folder.

#### Option B: Manual Dataset Creation
Your dataset should follow the LJSpeech format with a CSV file (`filename|raw_text|normalized_text`).

**Dataset Quality Requirements:**
- Sample rate: 16kHz, 22.05kHz, or 44.1kHz (will be resampled to 16kHz automatically)
- Format: WAV (mono or stereo - will be converted to mono automatically)
- Duration: 3-10 seconds per segment (optimal for TTS)
- Minimum total duration: 30+ minutes for basic training
- Audio quality: Clean, minimal background noise

### 2. Configuration

**Most Important Settings:**
```python
# In src/config.py

# --- Model Selection ---
is_turbo: bool = True  # True for Turbo, False for Normal.

# --- Training Strategy (NEW) ---
is_lora: bool = True   # True: Efficient LoRA training (Recommended for < 10h data)
                       # False: Full Fine-Tune (High VRAM, for massive datasets)

# If is_lora = True, these settings apply:
lora_r: int = 64
lora_alpha: int = 128
lora_target_modules = ["c_attn", "c_proj", "c_fc", "spkr_enc"]
lora_modules_to_save = ["text_emb", "text_head"] # Auto-trains new vocab embeddings

# --- Vocabulary ---
new_vocab_size: int = 52260 if is_turbo else 2454 

# --- Dataset Format ---
ljspeech = False       # True for metadata.csv format
json_format = True     # True for JSON formatted datasets
preprocess = True      # Set to False ONLY if you already preprocessed the dataset
```

### 3. Start Training
```bash
python train.py
```
*   If `is_lora = True`, your trained model will be saved inside `chatterbox_output/new_lang_adapter/`.
*   If `is_lora = False`, your trained model will be saved as `chatterbox_output/t3_turbo_finetuned.safetensors` (or `t3_finetuned`).

---

## 🗣️ Inference & Packing (Speech Synthesis)

The inference script is smart. Based on your `src/config.py` settings, it will automatically detect whether you trained a LoRA adapter or a Full model and merge the weights on-the-fly.

### 1. Prepare Reference Audio (Prompt)
Place a clean, 3-10 second reference `.wav` file in `speaker_reference/reference.wav`.

### 2. Running Inference
Edit `inference.py` to set your desired text:
```python
TEXT_TO_SAY = "Merhaba, sesimi geliştirmem oldukça uzun zaman aldı ve şimdi sahip olduğuma göre, sessiz kalmayacağım."
AUDIO_PROMPT = "./speaker_reference/2.wav"
```
Run inference:
```bash
python inference.py
```
The output will be saved as `output.wav` (24kHz). The script automatically handles sentence splitting, concatenates pauses, and uses **Silero VAD** to trim silence/hallucinations at the end of generated audio.

### 3. Packing Your Model (For LoRA Users)
Once you have tested your LoRA using `inference.py` and are completely satisfied with the results, you should pack (merge) it into a single file for deployment or sharing.

```bash
python merge_lora.py
```
This process takes your base model and bakes the trained LoRA adapter directly into it. It will generate a standalone `t3_turbo_finetuned_merged.safetensors` file that you can use in production without needing PEFT or adapter folders anymore!

---

## 🛠️ Technical Details

### Why Preprocessing?
Original Chatterbox training pipelines often process audio "on-the-fly" (resampling, feature extraction) during training. This causes the GPU to wait for the CPU, slowing down training significantly.
By running `preprocess.py`, we:
1.  Extract Speaker Embeddings (Voice Encoder)
2.  Extract Acoustic Tokens (S3Gen)
3.  Tokenize Text
4.  Save everything as optimized PyTorch tensors (`.pt`)
This allows the `dataset.py` to simply load tensors, maximizing GPU utilization.

### Tokenizer Structure
**Turbo Model Tokenizer (Smart Vocab Extension):**
Turbo mode uses GPT-2's powerful BPE tokenizer as a base. The `setup.py` script performs a **"Vocab Extension"**: it intelligently adds all unique characters from our 23-language grapheme set to the GPT-2 vocabulary. This process ensures that:
1.  The model retains its powerful knowledge of English words and structures.
2.  Special characters from other languages (e.g., `ğ, ş, ı` for Turkish; `é, à, ç` for French) are recognized as single, whole tokens, dramatically improving learning efficiency.
3.  **You do not need to create a custom tokenizer manually.** The setup is fully automated.

### VAD Integration
During inference, `inference.py` uses Silero VAD to prevent hallucinations and sentence-ending elongations. This automatically trims unwanted silence and noise from generated audio.

### Audio Processing Pipeline
All audio processing uses **FFmpeg** for professional-quality results:
- **Input:** Automatic conversion to mono (1 channel)
- **Resampling:** Automatic resampling to required sample rates
- **Training:** 16kHz processing
- **Output:** 24kHz, 16-bit PCM WAV format

---

## 📝 Troubleshooting

**Error:** `RuntimeError: Error(s) in loading state_dict for T3... size mismatch`
*   **Solution:** `new_vocab_size` in config doesn't match the token count in `tokenizer.json`. Count the tokens in your json and update the config file.

**Error:** `FileNotFoundError: ... ve.safetensors`
*   **Solution:** You haven't downloaded base models. Run `python setup.py`.

**Error:** `CUDA out of memory`
*   **Solution:** Enable `is_lora = True`. If still OOM, reduce `BATCH_SIZE` in `src/config.py` and increase `grad_accum`. Gradient Checkpointing is already enabled by default.

**Poor Quality Output:**
*   Check reference audio quality (should be clean, at least 5 seconds).
*   Ensure adequate training data (minimum 30 minutes recommended).
*   If training the Turbo model, **switch to LoRA** as it is much more stable than Full Fine-Tuning.

---

## 🙏 Acknowledgments
Based on the Chatterbox TTS model architecture. Special thanks to the original authors and contributors.

## 📧 Support
For issues and questions, please review `src/config.py` options or open an issue on GitHub with detailed error logs.