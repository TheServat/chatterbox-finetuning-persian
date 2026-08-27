"""A browser front end for listening to what a checkpoint actually sounds like.

Everything learned about this model argues for the shape of this page.

Generation is stochastic, and at one fixed checkpoint the noise floor spanned
3.5x on the seed alone - so a single clip says almost nothing and the page draws
several at once, measures each, and marks the cleanest. Picking the best of a
handful is not a trick here; it is how the model is meant to be used.

The last checkpoint is not reliably the best one, so every checkpoint on disk is
selectable rather than assuming the newest wins.

Persian text is rewritten before the model sees it - digits spelled out, Arabic
letters folded, ezafe heh rebuilt - and a surprise there looks like a model
fault from the outside, so the normalised text is shown next to the audio.

    python tools/ui.py                     pick a device automatically
    python tools/ui.py --device cpu        leave the GPU to training
    python tools/ui.py --share             a public link

Training usually owns the card. With less than 3.5 GB free this loads on the
CPU by default and says so: slower per clip, but it does not interrupt a run.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import compat  # noqa: E402
from src.config import TrainConfig  # noqa: E402
from src.persian.normalize import normalize  # noqa: E402
from src.utils import noise_floor, normalise_peak, trim_onset_artifact  # noqa: E402

NEEDED_VRAM_GB = 3.5
LONG_TEXT_CHARS = 220


def measured_floors() -> dict[int, float]:
    """Noise-floor medians already measured for each checkpoint during training.

    Every checkpoint that drew samples logged the median over three draws. That
    is the one number measured for every checkpoint without anyone listening,
    so it is what the ranking rests on - and it says how clean a clip is, not
    how well it reads, which is why the label names it rather than calling a
    checkpoint "best" outright.
    """
    pattern = re.compile(
        r"checkpoint-(\d+): noise floor median ([\d.]+)")
    floors: dict[int, float] = {}
    for log in list(ROOT.glob("supervised_train*.log")) + list(ROOT.glob("local_train*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for step, value in pattern.findall(text):
            floors[int(step)] = float(value)
    return floors


def checkpoints() -> list[tuple[str, Path, float | None]]:
    """(label, path, measured floor) for every usable checkpoint.

    Ranked by what was measured, cleanest first, with unmeasured ones after -
    the newest checkpoint has usually not drawn samples yet, and "newest" is
    not the same as "best": on the previous run step 3500 beat the final 4256
    on every measure taken.
    """
    from tools.verify_checkpoint import inspect, is_good

    floors = measured_floors()
    output = Path(TrainConfig().output_dir)
    usable, seen = [], set()
    # keep/ first: tools/keep_best.py copies good checkpoints there precisely
    # because rotation would otherwise delete them, so that is where the best
    # ones live once training has moved on.
    for directory in (output / "keep", output):
        for path in sorted(directory.glob("checkpoint-*")):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if not (path.is_dir() and match and (path / "model.safetensors").exists()):
                continue
            step = int(match.group(1))
            if step in seen:
                continue
            if is_good(inspect(path)):
                seen.add(step)
                usable.append((step, path, floors.get(step)))

    if not usable:
        return []

    newest_step = max(step for step, _, _ in usable)
    scored = sorted((r for r in usable if r[2] is not None), key=lambda r: r[2])
    unscored = sorted((r for r in usable if r[2] is None), key=lambda r: -r[0])

    out = []
    for rank, (step, path, floor) in enumerate(scored):
        note = "cleanest measured" if rank == 0 else f"noise {floor:.5f}"
        if step == newest_step:
            note += ", newest"
        out.append((f"checkpoint-{step}  ({note})", path, floor))
    for step, path, _ in unscored:
        note = "newest, not measured yet" if step == newest_step else "not measured"
        out.append((f"checkpoint-{step}  ({note})", path, None))
    return out


TAGS = {
    "Sounds": ["laughter", "giggle", "guffaw", "sigh", "gasp", "groan", "cry",
               "whisper", "mumble", "cough", "sneeze", "sniff", "snore",
               "clear_throat", "inhale", "exhale", "kiss", "shhh"],
    "Hesitation": ["UH", "UM"],
    "Other": ["singing", "humming", "whistle", "music"],
    "Switch language": ["fa", "en", "ar", "tr", "fr", "de", "es", "it", "ru"],
}

TAG_NOTE = (
    "These are real tokens - `[laughter]` is token 607, not ten letters - and "
    "the multilingual base was trained with them. This Persian finetune was "
    "not: all 95,802 clips in its corpus contain zero tags. So they may carry "
    "over from the base or may not, and the only way to find out is to try one."
)


def training_is_running() -> bool:
    """Whether a trainer owns the card right now.

    Free VRAM is the wrong question: it swings by gigabytes within a single
    step, so a snapshot can read 5 GB free and then have nothing left by the
    time a model is loaded. Whether the process exists does not flicker.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
             "Where-Object { $_.CommandLine -like '*train.py*' }).Count"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return out.isdigit() and int(out) > 0
    except Exception:
        return False


def pick_device(requested: str) -> tuple[str, str]:
    """(device, why) - the GPU only when nothing else is using it."""
    if requested != "auto":
        return requested, f"asked for {requested}"
    if not torch.cuda.is_available():
        return "cpu", "no CUDA device"
    if training_is_running():
        return "cpu", "training has the card; --device cuda to override"
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 2**30
    if free_gb < NEEDED_VRAM_GB:
        return "cpu", f"only {free_gb:.1f} GB free of {total / 2**30:.0f}"
    return "cuda", f"{free_gb:.1f} GB free"


class Voice:
    """The engine, loaded once, with a checkpoint that can be swapped."""

    def __init__(self, device: str):
        self.device = torch.device(device)
        self.config = TrainConfig()
        self.loaded: str | None = None

        compat.apply()
        from src.engine import build_engine
        from src.model import ChatterboxTrainerWrapper
        from train import apply_lora

        print("building the engine ...", flush=True)
        self.engine = build_engine(self.config, device="cpu", for_training=True)
        self.engine.t3 = apply_lora(self.config, self.engine.t3)
        self.wrapper = ChatterboxTrainerWrapper(self.engine.t3)

    def load(self, checkpoint: str) -> str:
        if checkpoint == self.loaded:
            return f"checkpoint-{Path(checkpoint).name} already loaded"

        from safetensors.torch import load_file

        started = time.time()
        state = load_file(Path(checkpoint) / "model.safetensors")
        result = self.wrapper.load_state_dict(state, strict=False)
        adapter_missing = [k for k in result.missing_keys
                           if "lora" in k or "modules_to_save" in k]
        if adapter_missing:
            return (f"refusing {Path(checkpoint).name}: {len(adapter_missing)} "
                    "adapter tensors did not load, the audio would be meaningless")

        self.engine.t3.to(self.device).eval()
        self.engine.s3gen.to(self.device).eval()
        self.engine.ve.to(self.device).eval()
        self.engine.device = self.device
        compat.use_eager_attention(self.engine.t3)
        self.loaded = checkpoint
        return (f"{Path(checkpoint).name} loaded in {time.time() - started:.0f}s "
                f"on {self.device.type}")

    def say(self, text, reference, draws, seed, temperature, cfg_weight,
            exaggeration, repetition_penalty, min_p, top_p, long_form,
            chunk_chars, gap_seconds):
        """Returns (list of (path, label), normalised text, a report)."""
        spoken = normalize(text)
        if not spoken.strip():
            return [], "", "nothing to say"

        prompt = str(reference) if reference else self.config.inference_prompt_path
        settings = dict(temperature=float(temperature),
                        cfg_weight=float(cfg_weight),
                        exaggeration=float(exaggeration),
                        repetition_penalty=float(repetition_penalty),
                        min_p=float(min_p), top_p=float(top_p))

        out_dir = ROOT / "chatterbox_output" / "ui"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")

        results, rows = [], []
        for draw in range(int(draws)):
            torch.manual_seed(int(seed) + draw)
            started = time.time()
            with torch.no_grad():
                if long_form:
                    wav = self.engine.generate_long(
                        text, audio_prompt_path=prompt,
                        max_chunk_chars=int(chunk_chars),
                        gap_seconds=float(gap_seconds),
                        language_id=self.config.language_id, **settings)
                else:
                    wav = self.engine.generate(
                        text, language_id=self.config.language_id,
                        audio_prompt_path=prompt, **settings)
            took = time.time() - started

            audio = normalise_peak(trim_onset_artifact(
                wav.squeeze().cpu().numpy(), self.engine.sr))
            floor = noise_floor(audio, self.engine.sr)
            path = out_dir / f"{stamp}_draw{draw + 1}.wav"
            import soundfile as sf
            sf.write(path, audio, self.engine.sr)

            results.append((str(path), floor, len(audio) / self.engine.sr, took))
            compat.drop_inference_cache(self.engine.t3)

        quietest = min(range(len(results)), key=lambda i: results[i][1])
        audio_out = []
        for i, (path, floor, seconds, took) in enumerate(results):
            mark = "  ← cleanest" if i == quietest else ""
            audio_out.append((path, f"draw {i + 1}   {seconds:.1f}s   "
                                    f"noise {floor:.5f}{mark}"))
            rows.append(f"draw {i + 1}: {seconds:.1f}s, noise floor {floor:.5f}, "
                        f"{took:.0f}s to make")

        floors = sorted(r[1] for r in results)
        spread = floors[-1] / max(floors[0], 1e-9)
        report = "\n".join(rows)
        if len(results) > 1:
            report += (f"\n\nspread across draws: {spread:.1f}x. Same weights and "
                       "same text - only the sampling differs, which is why more "
                       "than one draw is worth taking.")
        return audio_out, spoken, report


def build(voice: Voice, device_note: str):
    import gradio as gr

    available = checkpoints()
    if not available:
        raise SystemExit("no usable checkpoint under chatterbox_output/")
    choices = [(label, str(path)) for label, path, _ in available]
    default = choices[0][1]
    print(voice.load(default), flush=True)

    with gr.Blocks(title="Persian Chatterbox") as page:
        measured = sum(1 for _, _, floor in available if floor is not None)
        gr.Markdown(
            f"### Persian Chatterbox — {device_note}\n\n"
            f"{len(available)} usable checkpoints, {measured} with a measured "
            "noise floor. The first is the cleanest of those, which is not the "
            "same as the best-sounding — that is what your ears are for."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Text", lines=4,
                    value=TrainConfig().inference_test_text,
                    rtl=True,
                )
                with gr.Row():
                    checkpoint = gr.Dropdown(
                        choices, value=default, scale=3,
                        label="Checkpoint — ranked by measured noise floor",
                    )
                    draws = gr.Slider(1, 5, value=3, step=1, label="Draws", scale=1)
                stock = sorted(
                    str(p) for p in (ROOT / "speaker_reference").glob("*.wav")
                )
                stock_choice = gr.Dropdown(
                    ["(use the recording or upload below)", *stock],
                    value=stock[0] if stock else "(use the recording or upload below)",
                    label="Reference voice on disk",
                )
                reference = gr.Audio(
                    label="…or record / upload one",
                    type="filepath", sources=["upload", "microphone"],
                    format="wav",
                )
                speak = gr.Button("Speak", variant="primary")

                with gr.Accordion("Tags — untested in Persian, see note",
                                  open=False):
                    gr.Markdown(TAG_NOTE)
                    tag_buttons = []
                    for group, names in TAGS.items():
                        gr.Markdown(f"**{group}**")
                        for row_start in range(0, len(names), 6):
                            with gr.Row():
                                for name in names[row_start:row_start + 6]:
                                    button = gr.Button(f"[{name}]", size="sm")
                                    tag_buttons.append((button, f"[{name}]"))

            with gr.Column(scale=2):
                gr.Markdown("**Generation**")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                        label="Temperature — higher wanders more")
                cfg_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                       label="CFG weight — how hard it follows the text")
                exaggeration = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Exaggeration — the emotion_adv conditioning, "
                          "trained; 0.5 is neutral")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05,
                                               label="Repetition penalty")
                with gr.Accordion("Sampling and long text", open=False):
                    min_p = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="min_p")
                    top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
                    seed = gr.Number(value=1234, precision=0, label="Seed")
                    long_form = gr.Checkbox(
                        value=False,
                        label="Long text — split on sentences, never mid-word")
                    chunk_chars = gr.Slider(80, 400, value=200, step=10,
                                            label="Characters per chunk")
                    gap_seconds = gr.Slider(0.0, 0.6, value=0.12, step=0.02,
                                            label="Gap between chunks (s)")

        status = gr.Markdown()
        gallery = gr.Dataset(components=[gr.Audio(visible=False)], visible=False)
        audio_slots = [gr.Audio(label=f"draw {i + 1}", visible=False)
                       for i in range(5)]
        spoken = gr.Textbox(label="What the model actually receives", lines=2,
                            rtl=True, interactive=False)
        report = gr.Textbox(label="Measured", lines=6, interactive=False)

        def on_checkpoint(path):
            return voice.load(path)

        def on_speak(text, stock_pick, recorded, *rest):
            # A recording wins when there is one; otherwise the file on disk.
            # Saying which was used matters - a microphone that silently
            # captured nothing is otherwise indistinguishable from one that
            # worked, and the audio just sounds like the default voice.
            chosen = recorded or (stock_pick if stock_pick and
                                  not stock_pick.startswith("(") else None)
            clips, normalised, note = voice.say(text, chosen, *rest)
            source = ("your recording" if recorded else
                      Path(chosen).name if chosen else "the default reference")
            note = f"reference voice: {source}\n\n{note}"
            updates = []
            for i, slot in enumerate(audio_slots):
                if i < len(clips):
                    path, label = clips[i]
                    updates.append(gr.update(value=path, label=label, visible=True))
                else:
                    updates.append(gr.update(visible=False))
            return [*updates, normalised, note, ""]

        def append_tag(tag, current):
            spaced = (current or "").rstrip()
            return (spaced + " " + tag + " ") if spaced else (tag + " ")

        for button, tag in tag_buttons:
            button.click(lambda current, t=tag: append_tag(t, current),
                         [text], [text])

        checkpoint.change(on_checkpoint, [checkpoint], [status])
        speak.click(
            on_speak,
            [text, stock_choice, reference, draws, seed, temperature, cfg_weight,
             exaggeration, repetition_penalty, min_p, top_p, long_form,
             chunk_chars, gap_seconds],
            [*audio_slots, spoken, report, status],
        )

    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    device, why = pick_device(args.device)
    note = f"running on {device} ({why})"
    print(note, flush=True)
    if device == "cpu":
        print("  a clip takes a while on the CPU; --device cuda once training "
              "is done", flush=True)

    page = build(Voice(device), note)
    # Without this the generated wavs live outside gradio's own temp directory,
    # and it refuses to serve them - which arrives as "Response content shorter
    # than Content-Length" and a player showing 0:00 that will not play.
    page.launch(server_port=args.port, share=args.share, inbrowser=True,
                allowed_paths=[str(ROOT / "chatterbox_output"),
                               str(ROOT / "speaker_reference")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
