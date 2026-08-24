from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]


def _supports_bf16() -> bool:
    """True only on hardware with native bfloat16 units.

    Compute capability is the reliable signal: bf16 arrives with Ampere (8.0).
    Turing (RTX 20xx, Quadro RTX, T4) is 7.5 and has no bf16 hardware.

    `torch.cuda.is_bf16_supported()` cannot be used on its own - since torch 2.x
    it counts *emulated* bf16 and returns True on Turing, where training then
    runs correctly but slowly. That is a silent performance cliff rather than an
    error, which is exactly why it is worth checking properly.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return True
        try:
            return torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            return False  # older torch: no way to exclude emulation, assume no
    except Exception:
        return False


@dataclass
class TrainConfig:
    # --- Language mode ---
    # Persian is not one of Chatterbox's 23 built-in languages, so this mode
    # starts from the multilingual base, uses the [fa]-extended tokenizer, and
    # normalises text through src/persian/normalize.py.
    is_persian: bool = True
    language_id: str = "fa"

    # --- Paths ---
    model_dir: str = "./pretrained_models"

    # LJSpeech layout: metadata.csv is `id|raw-text|normalized-text` with no
    # header, wavs/ holds one file per row. Build it with tools/build_dataset.py.
    csv_path: str = "./MyTTSDataset/metadata.csv"
    wav_dir: str = "./MyTTSDataset/wavs"
    preprocessed_dir: str = "./MyTTSDataset/preprocess"

    output_dir: str = "./chatterbox_output"

    is_inference: bool = False
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = (
        "سلام، این یک آزمایش برای مدل گفتار فارسی است. "
        "امیدوارم صدای طبیعی و روانی داشته باشد."
    )

    # --- Dataset format ---
    ljspeech: bool = True    # True for the LJSpeech layout above
    json_format: bool = False
    preprocess: bool = True  # set False to reuse an existing preprocess/ cache

    # --- Base model ---
    # Persian builds on the multilingual checkpoint, which already knows Arabic
    # script from its Arabic and Hebrew training. The English base (704 tokens)
    # has never seen the script at all and would have to learn it from scratch.
    is_turbo: bool = False   # Turbo uses a GPT-2 tokenizer: poor fit for Persian
    is_lora: bool = True     # LoRA for < ~100 h; full finetune needs far more VRAM

    t3_filename: str = "t3_mtl23ls_v3.safetensors"
    s3gen_filename: str = "s3gen_v3.pt"
    ve_filename: str = "ve.safetensors"
    tokenizer_filename: str = "tokenizer_fa.json"

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    turbo_lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "c_fc", "spkr_enc"]
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "spkr_enc",
        ]
    )
    # The embedding and output head must train in full: [fa] is a brand-new row
    # and the Persian character rows have only ever been trained for Arabic.
    lora_modules_to_save: List[str] = field(
        default_factory=lambda: ["text_emb", "text_head"]
    )

    # --- Hyperparameters ---
    batch_size: int = 4
    grad_accum: int = 8      # effective batch = batch_size * grad_accum
    learning_rate: float = 1e-4
    num_epochs: int = 5
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    save_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 5
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True

    # Voice-conditioning dropout: how often the speaker embedding and prompt are
    # zeroed during training. Without it the model leans on the reference audio
    # and cannot speak from text alone.
    cond_dropout: float = 0.20
    seed: int = 1234

    # --- Constraints ---
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_len: int = 256
    max_speech_len: int = 850    # ~34 s at the 25 Hz speech-token rate
    prompt_duration: float = 3.0

    # --- Vocabulary ---
    @property
    def new_vocab_size(self) -> int:
        """Text-token table size this run trains with."""
        if self.is_turbo:
            return 52260         # GPT-2 medium merged with the expanded vocab
        if self.is_persian:
            return 2455          # multilingual 2454 + [fa]
        return 2454              # multilingual as shipped

    # --- Precision ---
    @property
    def precision(self) -> str:
        """`bf16`, `fp16` or `fp32`, chosen from what the GPU actually supports."""
        if _supports_bf16():
            return "bf16"
        try:
            import torch

            return "fp16" if torch.cuda.is_available() else "fp32"
        except Exception:
            return "fp32"

    @property
    def use_bf16(self) -> bool:
        return self.precision == "bf16"

    @property
    def use_fp16(self) -> bool:
        return self.precision == "fp16"

    # --- Resolved paths ---
    def path(self, name: str) -> Path:
        return (_ROOT / name).resolve() if not Path(name).is_absolute() else Path(name)

    @property
    def tokenizer_path(self) -> Path:
        return self.path(self.model_dir) / self.tokenizer_filename

    def describe(self) -> str:
        mode = "PERSIAN (multilingual base)" if self.is_persian else (
            "TURBO" if self.is_turbo else "ENGLISH"
        )
        strategy = f"LoRA r={self.lora_r}" if self.is_lora else "full finetune"
        return (
            f"mode={mode}  strategy={strategy}  vocab={self.new_vocab_size}  "
            f"precision={self.precision}  "
            f"batch={self.batch_size}x{self.grad_accum}"
            f"={self.batch_size * self.grad_accum}"
        )
