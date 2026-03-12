from __future__ import annotations

"""
quality_gate.py (Quality -> Gate to next analytical level)

Responsibility:
- Consume the per-measurement scores CSV produced by QualityRunner
  (quality_scores_by_file.csv) and decide PASS/FAIL using a threshold.
- Produce minimal artifacts to chain the pipeline:
    - quality_gate.csv
    - pass_mids.csv
    - fail_mids.csv
    - informational_queue.csv

This module does NOT:
- generate plots
- perform Level 2 informational analysis
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List

import pandas as pd


# -----------------------------------------------------------------------------
# Policy / artifacts
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GatePolicy:
    """Gate policy."""
    pass_score: float = 7.0
    overwrite: bool = True

    # If these columns exist, enforce minimum valid channels (recommended).
    # This keeps the gate consistent with the runner when available.
    require_min_channels_if_available: bool = True


@dataclass(frozen=True)
class QualityGateArtifacts:
    output_dir: Path
    gate_csv: Path
    pass_mids_csv: Path
    fail_mids_csv: Path
    informational_queue_csv: Path


# -----------------------------------------------------------------------------
# Gate
# -----------------------------------------------------------------------------

class QualityGate:
    """
    Quality gate (PASS/FAIL) to open the door to Level 2.

    Typical input:
      <target_root>/Reports/Level1_Quality/quality_scores_by_file.csv

    Minimal required columns:
      - mid
      - score_medicion

    Additional columns are preserved (fecha, lab, etiqueta, parquet_path, etc.).
    Legacy fields such as turno/jornada are optional and only kept if present.
    """

    def __init__(self, policy: Optional[GatePolicy] = None) -> None:
        self.policy = policy or GatePolicy()

    def run(
        self,
        *,
        scores_by_file_csv: Path,
        output_dir: Path,
        pass_score: Optional[float] = None,
        verbose: bool = True,
    ) -> QualityGateArtifacts:
        scores_by_file_csv = Path(scores_by_file_csv)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not scores_by_file_csv.exists():
            raise FileNotFoundError(f"scores_by_file_csv not found: {scores_by_file_csv}")

        df = pd.read_csv(scores_by_file_csv)

        # Minimal validation
        required_cols = ("mid", "score_medicion")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Scores CSV missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        thr = float(self.policy.pass_score if pass_score is None else pass_score)

        # Normalize score to float
        df["score_medicion"] = pd.to_numeric(df["score_medicion"], errors="coerce")
        if df["score_medicion"].isna().all():
            raise ValueError("score_medicion could not be converted to numeric (all NaN).")

        # Compute PASS/FAIL by threshold
        pass_mask = df["score_medicion"] >= thr

        # Optional: enforce minimum valid channels if runner fields exist
        if self.policy.require_min_channels_if_available:
            if ("n_channels_used" in df.columns) and ("min_channels_required" in df.columns):
                n_used = pd.to_numeric(df["n_channels_used"], errors="coerce").fillna(0).astype(int)
                n_req = pd.to_numeric(df["min_channels_required"], errors="coerce").fillna(1).astype(int)
                pass_mask = pass_mask & (n_used >= n_req)

        df["pass_to_level2"] = pass_mask
        df["threshold_pass"] = thr

        # Artifacts
        gate_csv = output_dir / "quality_gate.csv"
        pass_mids_csv = output_dir / "pass_mids.csv"
        fail_mids_csv = output_dir / "fail_mids.csv"
        informational_queue_csv = output_dir / "informational_queue.csv"

        if not self.policy.overwrite:
            for p in (gate_csv, pass_mids_csv, fail_mids_csv, informational_queue_csv):
                if p.exists():
                    raise FileExistsError(f"File already exists and overwrite=False: {p}")

        # Write gate detail (preserve all columns)
        df.to_csv(gate_csv, index=False, encoding="utf-8")

        # PASS/FAIL lists
        pass_mids = (
            df.loc[df["pass_to_level2"], ["mid"]]
            .dropna()
            .drop_duplicates()
            .astype({"mid": "string"})
            .sort_values("mid")
        )
        fail_mids = (
            df.loc[~df["pass_to_level2"], ["mid"]]
            .dropna()
            .drop_duplicates()
            .astype({"mid": "string"})
            .sort_values("mid")
        )

        pass_mids.to_csv(pass_mids_csv, index=False, encoding="utf-8")
        fail_mids.to_csv(fail_mids_csv, index=False, encoding="utf-8")

        # Informational queue for Level 2 (keep only useful cols if present)
        preferred_cols: List[str] = [
            "mid",
            "score_medicion",
            "score_mean_valid_channels",
            "threshold_pass",
            "pass_to_level2",
            "status",
            "fecha",
            "lab",
            "turno",
            "jornada",
            "etiqueta",
            "archivo",
            "color",
            "parquet_path",
            "n_channels_total",
            "n_channels_used",
            "n_channels_invalid",
            "min_channels_required",
        ]
        keep_cols = [c for c in preferred_cols if c in df.columns]

        queue = df.loc[df["pass_to_level2"], keep_cols].copy()

        # Stable sorting when metadata exists
        sort_cols: Sequence[str] = [c for c in ["fecha", "lab", "mid", "turno", "jornada"] if c in queue.columns]
        if sort_cols:
            queue = queue.sort_values(list(sort_cols))
        else:
            queue = queue.sort_values(["mid"]) if "mid" in queue.columns else queue

        queue.to_csv(informational_queue_csv, index=False, encoding="utf-8")
        #
        if verbose:
            n = int(df.shape[0])
            n_pass = int(df["pass_to_level2"].sum()) if n else 0
            n_fail = n - n_pass
            print(f"[QualityGate] Threshold: {thr:.2f} | Files scored: {n} | PASS: {n_pass} | FAIL: {n_fail}")

            if "lab" in df.columns and n > 0:
                lab_summary = (
                    df.groupby("lab", dropna=False)["pass_to_level2"]
                    .agg(["count", "sum"])
                    .reset_index()
                    .rename(columns={"count": "n_files", "sum": "n_pass"})
                )
                lab_summary["n_fail"] = lab_summary["n_files"] - lab_summary["n_pass"]
                print("[QualityGate] PASS/FAIL by lab:")
                for r in lab_summary.itertuples(index=False):
                    print(f"  - {r.lab}: files={int(r.n_files)} | pass={int(r.n_pass)} | fail={int(r.n_fail)}")

            print(f"[QualityGate] gate_csv: {gate_csv}")
            print(f"[QualityGate] pass_mids: {pass_mids_csv}")
            print(f"[QualityGate] fail_mids: {fail_mids_csv}")
            print(f"[QualityGate] informational_queue: {informational_queue_csv}")
        return QualityGateArtifacts(
            output_dir=output_dir,
            gate_csv=gate_csv,
            pass_mids_csv=pass_mids_csv,
            fail_mids_csv=fail_mids_csv,
            informational_queue_csv=informational_queue_csv,
        )


__all__ = [
    "GatePolicy",
    "QualityGateArtifacts",
    "QualityGate",
]
