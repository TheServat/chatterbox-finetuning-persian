"""Finetune Chatterbox's T3 for Persian.

Only T3 - the model that turns text into S3 speech tokens - is trained. The
S3Gen decoder and the voice encoder are frozen, which is right for adding a
language: S3Gen turns speech tokens into a waveform and knows nothing about
which language produced them.

The Persian path starts from the multilingual checkpoint rather than the
English one. The English base has 704 text tokens and has never seen Arabic
script; the multilingual base already carries all 2454 grapheme tokens and real
Arabic and Hebrew training behind them, so Persian starts from something close
instead of from noise. `src/engine.py` builds it, grows the text embedding by
one row for `[fa]`, and seeds that row from `[ar]`.

    python tools/fetch_models.py      # collect the weights
    python tools/build_dataset.py     # build MyTTSDataset/
    python train.py
"""

import os
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from src import compat  # noqa: F401  (adds T3's training hooks)
from src.config import TrainConfig
from src.dataset import (
    ChatterboxDataset,
    data_collator_standart,
    data_collator_turbo,
)
from src.engine import build_engine
from src.inference_callback import InferenceCallback
from src.model import ChatterboxTrainerWrapper
from src.preprocess_file_based import preprocess_dataset_file_based
from src.preprocess_json import preprocess_dataset_json_based
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.utils import setup_logger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = setup_logger("ChatterboxFinetune")


def check_inputs(cfg: TrainConfig) -> bool:
    """Fail before loading 3 GB of weights, not after."""
    problems = []

    model_dir = cfg.path(cfg.model_dir)
    for filename in (
        cfg.t3_filename, cfg.s3gen_filename, cfg.ve_filename, cfg.tokenizer_filename
    ) if cfg.is_persian else ():
        if not (model_dir / filename).exists():
            problems.append(f"missing {model_dir / filename}")

    if cfg.preprocess and not cfg.path(cfg.csv_path).exists():
        problems.append(
            f"missing {cfg.csv_path} - build it with `python tools/build_dataset.py`"
        )

    if problems:
        logger.error("Cannot start:")
        for problem in problems:
            logger.error(f"  - {problem}")
        logger.error("Run `python tools/fetch_models.py` to collect the weights.")
        return False
    return True


def apply_lora(cfg: TrainConfig, t3):
    """Wrap T3 in LoRA, keeping the text embedding and head fully trainable.

    Those two are not adapters-and-freeze material: `[fa]` is a brand-new row,
    and every Persian character row has so far only ever been trained to sound
    Arabic. A low-rank update on top of them would not be enough.
    """
    from peft import LoraConfig, get_peft_model

    for param in t3.parameters():
        param.requires_grad = False

    targets = (
        cfg.turbo_lora_target_modules if cfg.is_turbo else cfg.lora_target_modules
    )
    logger.info(f"LoRA targets: {targets}")
    logger.info(f"Fully trained: {cfg.lora_modules_to_save}")

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=targets,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        modules_to_save=cfg.lora_modules_to_save,
    )
    model = get_peft_model(t3, peft_config)
    model.print_trainable_parameters()
    return model


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--max-steps", type=int,
        help="stop after N optimiser steps - use a small value to smoke-test "
             "the whole path (model, LoRA, collator, backward) in a minute",
    )
    parser.add_argument("--epochs", type=float, help="override num_epochs")
    parser.add_argument("--batch-size", type=int, help="override batch_size")
    parser.add_argument("--grad-accum", type=int, help="override grad_accum")
    parser.add_argument("--lora-r", type=int, help="override lora_r")
    parser.add_argument("--lr", type=float, help="override learning_rate")
    parser.add_argument("--workers", type=int, help="override dataloader workers")
    parser.add_argument("--save-steps", type=int, help="override save_steps")
    parser.add_argument("--output-dir", help="override output_dir")
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="reuse the existing preprocess/ cache",
    )
    parser.add_argument(
        "--status-file",
        help="write structured progress here (default: <output_dir>/status.json), "
             "which is where tools/watch.py looks for it",
    )
    parser.add_argument(
        "--hourly-rate", type=float, default=0.0,
        help="GPU price per hour, so the status file can report cost so far",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from the newest checkpoint in output_dir - the way to "
             "survive a Colab disconnect without losing the run",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="synthesise a Persian sample at every checkpoint, so progress can "
             "be heard rather than inferred from the loss curve",
    )
    parser.add_argument(
        "--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
        help="override the detected precision (auto is almost always right)",
    )
    return parser.parse_args(argv)


def configure(args) -> TrainConfig:
    cfg = TrainConfig()

    for attr, value in (
        ("num_epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("grad_accum", args.grad_accum),
        ("lora_r", args.lora_r),
        ("learning_rate", args.lr),
        ("dataloader_num_workers", args.workers),
        ("save_steps", args.save_steps),
        ("output_dir", args.output_dir),
    ):
        if value is not None:
            setattr(cfg, attr, value)

    if args.no_preprocess:
        cfg.preprocess = False

    # Always write a status file. It costs nothing, and a run nobody can see the
    # progress of is a run nobody notices has stalled.
    if not args.status_file:
        args.status_file = str(Path(cfg.output_dir) / "status.json")

    if args.sample:
        cfg.is_inference = True

    if args.lora_r is not None:
        # alpha tracks r; changing one without the other silently rescales the
        # effective learning rate of every adapter.
        cfg.lora_alpha = args.lora_r * 2

    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = configure(args)

    logger.info("--- Chatterbox finetuning ---")
    logger.info(cfg.describe())
    if args.max_steps:
        logger.info(f"Smoke test: stopping after {args.max_steps} steps")

    if not check_inputs(cfg):
        return 1

    precision = cfg.precision if args.precision == "auto" else args.precision
    if precision == "fp16":
        logger.warning(
            "This GPU has no native bf16 (pre-Ampere), so training runs in fp16. "
            "It works, but bf16 is more forgiving of loss spikes - prefer an "
            "Ampere or newer card for the full run."
        )

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Built on CPU: the engine loads ~3 GB and preprocessing wants the VRAM.
    engine = build_engine(cfg, device="cpu", for_training=True)

    logger.info("Freezing S3Gen and the voice encoder")
    for module in (engine.ve, engine.s3gen):
        for param in module.parameters():
            param.requires_grad = False

    if cfg.preprocess:
        logger.info("Preprocessing...")
        if cfg.ljspeech:
            preprocess_dataset_ljspeech(cfg, engine)
        elif cfg.json_format:
            preprocess_dataset_json_based(cfg, engine)
        else:
            preprocess_dataset_file_based(cfg, engine)
        # Preprocessing put S3Gen and the voice encoder on the GPU; T3 needs
        # that memory now and neither is used again during training.
        engine.ve.to("cpu")
        engine.s3gen.to("cpu")
        torch.cuda.empty_cache()
    else:
        logger.info("Skipping preprocessing (cfg.preprocess is False)")

    if cfg.is_lora:
        engine.t3 = apply_lora(cfg, engine.t3)
    else:
        logger.info("Full finetune: every T3 parameter is trainable")
        engine.t3.train()
        for param in engine.t3.parameters():
            param.requires_grad = True

    train_dataset = ChatterboxDataset(cfg)
    model = ChatterboxTrainerWrapper(engine.t3)

    collator = data_collator_turbo if cfg.is_turbo else data_collator_standart
    logger.info(f"Collator: {'turbo' if cfg.is_turbo else 'standard'}")

    # The callback samples from this very engine, so the audio reflects the
    # weights at that step and no second copy competes for VRAM.
    callbacks = [InferenceCallback(cfg, engine=engine)] if cfg.is_inference else []

    status = None
    if args.status_file:
        from src.status_callback import StatusCallback

        status = StatusCallback(
            args.status_file,
            hourly_rate=args.hourly_rate,
            extra={"config": cfg.describe()},
        )
        callbacks.append(status)

    # transformers 5.2 deprecated warmup_ratio in favour of warmup_steps, so the
    # ratio is resolved here against the actual step count.
    steps_per_epoch = max(
        1, len(train_dataset) // max(1, cfg.batch_size * cfg.grad_accum)
    )
    total_steps = args.max_steps or int(steps_per_epoch * cfg.num_epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    logger.info(
        f"{steps_per_epoch} steps/epoch, {total_steps} total, "
        f"{warmup_steps} warmup"
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        remove_unused_columns=False,   # the wrapper takes its own field names
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_persistent_workers=cfg.dataloader_num_workers > 0,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        max_steps=args.max_steps if args.max_steps else -1,
        gradient_checkpointing=cfg.gradient_checkpointing,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    resume = False
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint

        last = get_last_checkpoint(cfg.output_dir) if os.path.isdir(cfg.output_dir) else None
        if last:
            logger.info(f"Resuming from {last}")
            resume = last
        else:
            logger.info(f"--resume: no checkpoint in {cfg.output_dir}, starting fresh")

    logger.info("Training...")
    try:
        trainer.train(resume_from_checkpoint=resume or None)
    except Exception as exc:
        # The watcher polls the status file; without this a crash looks
        # identical to a hang, and the pod keeps billing until a timeout.
        if status:
            status.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise

    if args.max_steps:
        peak = (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else 0
        )
        logger.info(
            f"Smoke test finished. Peak VRAM {peak:.2f} GB at "
            f"batch={cfg.batch_size} accum={cfg.grad_accum} "
            f"lora_r={cfg.lora_r} precision={precision}."
        )
        logger.info("Not saving: a few steps produce nothing worth keeping.")
        if status:
            status.finish("done")
        return 0

    logger.info("Saving...")
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.is_lora:
        save_path = os.path.join(cfg.output_dir, "persian_adapter")
        engine.t3.save_pretrained(save_path)
        logger.info(f"Adapter saved to {save_path}")
        logger.info(
            "It holds the LoRA weights and the resized text embedding, and is "
            f"tied to {cfg.t3_filename} - it will not load onto another base."
        )
    else:
        from safetensors.torch import save_file

        filename = (
            "t3_turbo_finetuned.safetensors"
            if cfg.is_turbo
            else "t3_fa_finetuned.safetensors"
        )
        path = os.path.join(cfg.output_dir, filename)
        save_file(engine.t3.state_dict(), path)
        logger.info(f"Full model saved to {path}")

    if status:
        status.finish("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
