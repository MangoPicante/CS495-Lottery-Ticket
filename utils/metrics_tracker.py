"""MetricsTracker — records training metrics and writes result artifacts.

Captures:
  - Wall-clock time (seconds since training start)
  - Peak RSS memory (MB) via psutil
  - Loss and perplexity per step
  - CO₂ emissions (kg) via CodeCarbon (optional — gracefully skipped if absent)

Outputs (written to output_dir/):
  - step_metrics.csv  — one row per logged step
  - run_summary.json  — final aggregate stats + model config
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import psutil


class MetricsTracker:
    """Records per-step metrics and saves results to disk."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._process = psutil.Process()
        self._start_time: Optional[float] = None
        self._step_metrics: list[dict[str, Any]] = []
        self._peak_memory_mb: float = 0.0

        # CodeCarbon is optional — import at runtime so the rest of the
        # codebase works even if the package is not installed.
        self._carbon_tracker = None
        try:
            from codecarbon import EmissionsTracker  # type: ignore[import]

            self._carbon_tracker = EmissionsTracker(
                output_dir=str(self.output_dir),
                project_name="bitnet-capstone",
                log_level="error",
                save_to_file=True,
            )
        except Exception:
            pass  # codecarbon not available

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin timing and carbon tracking."""
        self._start_time = time.time()
        if self._carbon_tracker is not None:
            try:
                self._carbon_tracker.start()
            except Exception:
                self._carbon_tracker = None

    def stop_and_save(self, model_config: dict[str, Any]) -> Path:
        """Stop tracking, write artifacts, return path to run_summary.json."""
        total_time = time.time() - (self._start_time or time.time())

        emissions_kg: Optional[float] = None
        if self._carbon_tracker is not None:
            try:
                emissions_kg = self._carbon_tracker.stop()
            except Exception:
                pass

        self._write_step_csv()

        final = self._step_metrics[-1] if self._step_metrics else {}
        summary: dict[str, Any] = {
            "model_config": model_config,
            "total_time_sec": round(total_time, 2),
            "total_steps": len(self._step_metrics),
            "final_loss": final.get("loss"),
            "final_perplexity": final.get("perplexity"),
            "peak_memory_mb": round(self._peak_memory_mb, 1),
            "co2_kg": emissions_kg,
        }

        summary_path = self.output_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    # ------------------------------------------------------------------
    # Per-step logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        loss: float,
        tokens_per_sec: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Record metrics for one optimizer step."""
        elapsed = time.time() - (self._start_time or time.time())
        mem_mb = self._process.memory_info().rss / 1_000_000

        if mem_mb > self._peak_memory_mb:
            self._peak_memory_mb = mem_mb

        # Guard against extreme loss values before exp()
        ppl = math.exp(min(loss, 20.0))

        entry: dict[str, Any] = {
            "step": step,
            "loss": round(loss, 6),
            "perplexity": round(ppl, 3),
            "elapsed_sec": round(elapsed, 2),
            "memory_mb": round(mem_mb, 1),
        }
        if tokens_per_sec is not None:
            entry["tokens_per_sec"] = round(tokens_per_sec, 1)
        if lr is not None:
            entry["lr"] = lr

        self._step_metrics.append(entry)

    def log_step_dict(self, d: dict[str, Any]) -> None:
        """Log a pre-built metrics dict (useful for resuming from checkpoint)."""
        self._step_metrics.append(d)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds since start()."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def peak_memory_mb(self) -> float:
        return self._peak_memory_mb

    @property
    def step_metrics(self) -> list[dict[str, Any]]:
        return list(self._step_metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_step_csv(self) -> None:
        if not self._step_metrics:
            return
        csv_path = self.output_dir / "step_metrics.csv"
        fieldnames = list(self._step_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._step_metrics)

    def __repr__(self) -> str:
        return (
            f"MetricsTracker(output_dir={self.output_dir}, "
            f"steps={len(self._step_metrics)}, "
            f"elapsed={self.elapsed:.1f}s)"
        )
