"""Write training progress to a JSON file the outside world can poll.

A remote run has to be watchable from a laptop that cannot SSH into the pod, so
progress needs to leave the trainer in a form something else can read. Parsing
stdout would work until a log format changes; a callback writing structured
state does not have that problem.

The file is rewritten atomically on every log event, so a reader never sees a
half-written document, and it carries enough to answer the questions that
actually matter mid-run: is it progressing, is the loss sane, how much longer,
and how much has it cost so far.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path

from transformers import TrainerCallback


class StatusCallback(TrainerCallback):
    """Mirror the trainer's progress into `status_path` as JSON."""

    def __init__(self, status_path: str | Path, *, hourly_rate: float = 0.0,
                 extra: dict | None = None):
        self.path = Path(status_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hourly_rate = hourly_rate
        self.extra = extra or {}

        self.started = time.time()
        # A cumulative rate is dragged down by minutes of model loading and
        # reads far below reality; a rolling window answers "how fast is it
        # going now", which is what an ETA and a stall check both need.
        self.recent: deque[tuple[int, float]] = deque(maxlen=12)
        self.state: dict = {
            "phase": "starting",
            "ok": True,
            "error": None,
            "started_at": self.started,
            **self.extra,
        }
        self._write()

    # -- lifecycle ---------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        self.state.update(
            phase="training",
            max_steps=state.max_steps,
            num_epochs=args.num_train_epochs,
            batch_size=args.per_device_train_batch_size,
            grad_accum=args.gradient_accumulation_steps,
        )
        self._write()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        elapsed = time.time() - self.started

        if "loss" in logs:
            loss = logs["loss"]
            # A NaN or infinite loss means the run is already dead; recording it
            # lets the watchdog stop the pod instead of paying for a dead run.
            if isinstance(loss, (int, float)) and not math.isfinite(loss):
                self.state.update(ok=False, error=f"loss is {loss}")
            self.state["loss"] = loss

        for key in ("learning_rate", "grad_norm", "epoch"):
            if key in logs:
                self.state[key] = logs[key]

        step = state.global_step
        self.state["step"] = step
        self.state["elapsed_seconds"] = round(elapsed, 1)

        self.recent.append((step, time.time()))
        if step > 0 and elapsed > 0:
            samples_per_step = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            self.state["samples_per_sec_average"] = round(
                step / elapsed * samples_per_step, 2
            )

            steps_per_sec = None
            if len(self.recent) >= 2:
                (first_step, first_time), (last_step, last_time) = (
                    self.recent[0], self.recent[-1]
                )
                window = last_time - first_time
                if window > 0 and last_step > first_step:
                    steps_per_sec = (last_step - first_step) / window
            if steps_per_sec is None:
                steps_per_sec = step / elapsed

            self.state["steps_per_sec"] = round(steps_per_sec, 4)
            self.state["samples_per_sec"] = round(steps_per_sec * samples_per_step, 2)

            if state.max_steps:
                remaining = (state.max_steps - step) / max(steps_per_sec, 1e-9)
                self.state["eta_seconds"] = round(remaining, 0)
                self.state["eta_hours"] = round(remaining / 3600, 2)
                self.state["progress"] = round(step / state.max_steps, 4)

        if self.hourly_rate:
            self.state["cost_so_far"] = round(elapsed / 3600 * self.hourly_rate, 3)

        self.state["gpu"] = _gpu_snapshot()
        self._write()

    def on_save(self, args, state, control, **kwargs):
        self.state["last_checkpoint_step"] = state.global_step
        self.state["last_checkpoint_at"] = time.time()
        self._write()

    def on_train_end(self, args, state, control, **kwargs):
        self.state.update(phase="finishing", step=state.global_step)
        self._write()

    # -- helpers -----------------------------------------------------------

    def finish(self, phase: str = "done", error: str | None = None) -> None:
        """Record the terminal state, so a watcher stops waiting."""
        self.state.update(phase=phase, error=error, ok=error is None,
                          finished_at=time.time())
        self._write()

    def _write(self) -> None:
        self.state["updated_at"] = time.time()
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.replace(temporary, self.path)


def _gpu_snapshot() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        # mem_get_info reports the whole card, not this process, so on a shared
        # GPU it silently includes everyone else. Recording what torch itself
        # holds alongside it is what makes the difference legible: an Ollama
        # model loaded mid-run took a 6 GB card to 95% and slowed training from
        # 7.4 to 30 s/step, and nothing in the logs said so.
        free, total = torch.cuda.mem_get_info()
        return {
            "name": torch.cuda.get_device_name(0),
            "mem_used_gb": round((total - free) / 2**30, 2),
            "mem_total_gb": round(total / 2**30, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
    except Exception:
        return None
