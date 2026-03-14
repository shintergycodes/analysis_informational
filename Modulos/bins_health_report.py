
# bins_health_report.py
from __future__ import annotations

"""

Global diagnostic for Level 4 bins ("bins_spec.json") health.

Goals
-----
- Validate the bins contract (schema, monotonic edges, sane supports).
- Quantify risk factors (micro-bins, extreme Q/K, narrow supports).
- Optionally estimate *clipping / out-of-range* rates by scanning a sampled set of
  analysis-ready parquets (low-resource friendly: read only laser columns).

Outputs
-------
- bins_health_report.json   (global + per-laser + rule outcomes)
- bins_health_per_laser.csv (flat table for quick inspection)

This module is intended to be called by main_v3.py immediately after bins are built,
before informational_states.py runs.

All code is written in English by design (project constraint).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import math
import numpy as np
import pandas as pd


# =============================================================================
# Exceptions
# =============================================================================

class BinsHealthError(RuntimeError):
    """Raised when the bins health evaluation cannot run."""


# =============================================================================
# Policy
# =============================================================================

@dataclass(frozen=True)
class BinsHealthPolicy:
    # Sampling/parquet scan
    scan_parquets: bool = True
    parquet_sample_n: int = 30          # number of parquets to scan (bounded)
    per_parquet_row_cap: int = 200_000   # cap rows read per parquet column (bounded)
    seed: int = 123456
    parquet_engine: str = "auto"         # "auto" uses pandas default

    # Thresholds (warn/fail)
    # Clipping: fraction of values outside support (before clipping)
    clip_warn: float = 0.005    # 0.5%
    clip_fail: float = 0.02     # 2%

    # NaNs / non-finite rate
    nan_warn: float = 0.001
    nan_fail: float = 0.01

    # Edge health
    min_bins_warn: int = 8
    min_bins_fail: int = 4

    # "Micro-bin" threshold: edges that are too close (relative to span)
    microbin_rel_warn: float = 1e-6
    microbin_rel_fail: float = 1e-8

    # Coupling Q (informative; not necessarily fail unless extreme)
    coupling_Q_warn: int = 24
    coupling_Q_fail: int = 64

    # Spectral K (informative)
    spectral_K_warn: int = 64
    spectral_K_fail: int = 128

    # Frequency support sanity checks
    spectral_allow_unitless: bool = True  # allow cycles_per_sample / fs=1.0


# =============================================================================
# Public API
# =============================================================================

def evaluate_bins_health(
    *,
    bins_spec_json: Union[str, Path],
    quality_scores_by_file_csv: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    policy: BinsHealthPolicy = BinsHealthPolicy(),
) -> Dict[str, Any]:
    """
    Evaluate bins health globally and per laser.

    Parameters
    ----------
    bins_spec_json:
        Path to bins_spec.json produced by informational_bins.py (schema_version "2.0").
    quality_scores_by_file_csv:
        Optional path to quality_scores_by_file.csv to discover parquet_path candidates.
        If None, only contract-level checks are performed (no clipping scan).
    out_dir:
        If provided, write:
            - bins_health_report.json
            - bins_health_per_laser.csv
    policy:
        Thresholds and scan parameters.

    Returns
    -------
    report: dict
        A JSON-serializable report.
    """
    bins_spec_json = Path(bins_spec_json)
    spec = _read_json(bins_spec_json)
    _validate_spec_v2(spec)

    channels = spec["channels"]
    lasers = [c["laser"] for c in channels]

    per_laser = []
    hard_errors: List[str] = []
    warnings: List[str] = []

    # --- Contract-only metrics (no data scan) ---
    for ch in channels:
        laser = ch["laser"]
        amp_edges = np.asarray(ch["amplitude_edges"], dtype=float)
        dx_edges = np.asarray(ch["increment_edges"], dtype=float)
        coup_edges = np.asarray(ch.get("coupling_edges", []), dtype=float) if "coupling_edges" in ch else None
        spec_edges = np.asarray(ch.get("spectral", {}).get("freq_edges", []), dtype=float) if "spectral" in ch else None

        amp_metrics = _edges_metrics(amp_edges)
        dx_metrics = _edges_metrics(dx_edges)

        coup_metrics = _edges_metrics(coup_edges) if coup_edges is not None and coup_edges.size > 0 else None
        spec_metrics = _edges_metrics(spec_edges) if spec_edges is not None and spec_edges.size > 0 else None

        row = {
            "laser": laser,
            "B_amp": int(amp_edges.size - 1),
            "B_dx": int(dx_edges.size - 1),
            "amp_span": float(amp_metrics["span"]),
            "dx_span": float(dx_metrics["span"]),
            "amp_microbin_frac": float(_microbin_fraction(amp_edges, amp_metrics["span"], policy)),
            "dx_microbin_frac": float(_microbin_fraction(dx_edges, dx_metrics["span"], policy)),
        }

        if coup_metrics is not None:
            row["Q_coupling"] = int(coup_edges.size - 1)
            row["coupling_span"] = float(coup_metrics["span"])
            row["coupling_microbin_frac"] = float(_microbin_fraction(coup_edges, coup_metrics["span"], policy))
        else:
            row["Q_coupling"] = None
            row["coupling_span"] = None
            row["coupling_microbin_frac"] = None

        if spec_metrics is not None:
            row["K_spectral"] = int(spec_edges.size - 1)
            row["spectral_span"] = float(spec_metrics["span"])
        else:
            row["K_spectral"] = None
            row["spectral_span"] = None

        per_laser.append(row)

        # Basic sanity warnings (contract level)
        if row["B_amp"] < policy.min_bins_warn:
            warnings.append(f"{laser}: low B_amp={row['B_amp']}")
        if row["B_dx"] < policy.min_bins_warn:
            warnings.append(f"{laser}: low B_dx={row['B_dx']}")
        if row.get("Q_coupling") is not None and row["Q_coupling"] >= policy.coupling_Q_warn:
            warnings.append(f"{laser}: high Q_coupling={row['Q_coupling']}")
        if row.get("K_spectral") is not None and row["K_spectral"] >= policy.spectral_K_warn:
            warnings.append(f"{laser}: high K_spectral={row['K_spectral']}")

    # --- Optional parquet scan for clipping / NaNs ---
    scan_stats = None
    if policy.scan_parquets and quality_scores_by_file_csv is not None:
        qpath = Path(quality_scores_by_file_csv)
        if not qpath.exists():
            raise BinsHealthError(f"quality_scores_by_file_csv not found: {qpath}")
        scan_stats = _scan_parquets_for_clip_rates(
            spec=spec,
            quality_scores_by_file_csv=qpath,
            policy=policy,
        )
        _merge_scan_stats_into_rows(per_laser, scan_stats)

        # Evaluate thresholds -> warnings/errors
        for r in per_laser:
            laser = r["laser"]
            for metric in ("clip_rate_amp", "clip_rate_dx", "clip_rate_coupling"):
                if metric not in r or r[metric] is None:
                    continue
                v = float(r[metric])
                if v >= policy.clip_fail:
                    hard_errors.append(f"{laser}: {metric}={v:.4%} (>= {policy.clip_fail:.2%})")
                elif v >= policy.clip_warn:
                    warnings.append(f"{laser}: {metric}={v:.4%} (>= {policy.clip_warn:.2%})")

            if "nan_rate_amp" in r and r["nan_rate_amp"] is not None:
                v = float(r["nan_rate_amp"])
                if v >= policy.nan_fail:
                    hard_errors.append(f"{laser}: nan_rate_amp={v:.4%} (>= {policy.nan_fail:.2%})")
                elif v >= policy.nan_warn:
                    warnings.append(f"{laser}: nan_rate_amp={v:.4%} (>= {policy.nan_warn:.2%})")

    # --- Global decision ---
    status = "OK"
    if hard_errors:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    report = {
        "status": status,
        "bins_spec": str(bins_spec_json),
        "schema_version": spec.get("schema_version"),
        "created_utc": spec.get("created_utc"),
        "mode": spec.get("mode"),
        "reference": spec.get("reference", {}),
        "policy": asdict(policy),
        "lasers": lasers,
        "per_laser": per_laser,
        "warnings": warnings,
        "errors": hard_errors,
        "scan": scan_stats,
    }

    # Write outputs if requested
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(report, out_dir / "bins_health_report.json")
        pd.DataFrame(per_laser).to_csv(out_dir / "bins_health_per_laser.csv", index=False, encoding="utf-8")

    return report


def print_bins_health_report(report: Dict[str, Any]) -> None:
    """Human-readable console output (main_v3 friendly)."""
    status = report.get("status", "UNKNOWN")
    print("\n" + "=" * 80)
    print("BINS HEALTH REPORT")
    print("=" * 80)
    print(f"Status: {status}")
    print(f"Bins spec: {report.get('bins_spec')}")
    print(f"Schema: {report.get('schema_version')} | Created: {report.get('created_utc')} | Mode: {report.get('mode')}")
    ref = report.get("reference", {}) or {}
    print(f"Reference: group={ref.get('group')} | n_parquets={ref.get('n_reference_parquets')} | rule={ref.get('selection_rule')}")
    print("-" * 80)

    # Compact per-laser table
    df = pd.DataFrame(report.get("per_laser", []))
    if not df.empty:
        cols = [c for c in [
            "laser", "B_amp", "B_dx", "Q_coupling", "K_spectral",
            "clip_rate_amp", "clip_rate_dx", "clip_rate_coupling",
            "nan_rate_amp"
        ] if c in df.columns]
        df2 = df[cols].copy()
        with pd.option_context("display.max_rows", 200, "display.width", 200):
            print(df2.to_string(index=False))

    warnings = report.get("warnings", []) or []
    errors = report.get("errors", []) or []

    if warnings:
        print("\nWARNINGS:")
        for w in warnings[:50]:
            print(f" - {w}")
        if len(warnings) > 50:
            print(f" ... ({len(warnings) - 50} more)")
    if errors:
        print("\nERRORS:")
        for e in errors[:50]:
            print(f" - {e}")
        if len(errors) > 50:
            print(f" ... ({len(errors) - 50} more)")

    print("=" * 80)


# =============================================================================
# Internal helpers
# =============================================================================

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise BinsHealthError(f"Failed to read JSON: {path} ({type(e).__name__}: {e})") from e


def _write_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_spec_v2(spec: Dict[str, Any]) -> None:
    if spec.get("schema_version") != "2.0":
        raise BinsHealthError(f"Unsupported schema_version: {spec.get('schema_version')!r}")
    if "channels" not in spec or not isinstance(spec["channels"], list) or not spec["channels"]:
        raise BinsHealthError("bins_spec missing non-empty 'channels' list.")

    for ch in spec["channels"]:
        for k in ("laser", "amplitude_edges", "increment_edges"):
            if k not in ch:
                raise BinsHealthError(f"Channel missing key {k!r}")
        for k in ("amplitude_edges", "increment_edges"):
            _ensure_edges_ok(np.asarray(ch[k], dtype=float), ctx=f"{ch['laser']}:{k}")
        if "coupling_edges" in ch:
            _ensure_edges_ok(np.asarray(ch["coupling_edges"], dtype=float), ctx=f"{ch['laser']}:coupling_edges")
        if "spectral" in ch and "freq_edges" in ch["spectral"]:
            _ensure_edges_ok(np.asarray(ch["spectral"]["freq_edges"], dtype=float), ctx=f"{ch['laser']}:spectral.freq_edges")


def _ensure_edges_ok(edges: np.ndarray, ctx: str) -> None:
    if edges.ndim != 1 or edges.size < 2:
        raise BinsHealthError(f"Invalid edges shape for {ctx}: {edges.shape}")
    if not np.all(np.isfinite(edges)):
        raise BinsHealthError(f"Non-finite edges in {ctx}")
    if not np.all(edges[1:] > edges[:-1]):
        raise BinsHealthError(f"Edges not strictly increasing in {ctx}")


def _edges_metrics(edges: np.ndarray) -> Dict[str, float]:
    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)
    span = float(edges[-1] - edges[0])
    return {
        "span": span,
        "min_width": float(np.min(widths)) if widths.size else float("nan"),
        "median_width": float(np.median(widths)) if widths.size else float("nan"),
        "max_width": float(np.max(widths)) if widths.size else float("nan"),
    }


def _microbin_fraction(edges: np.ndarray, span: float, policy: BinsHealthPolicy) -> float:
    edges = np.asarray(edges, dtype=float)
    if edges.size < 3:
        return 0.0
    widths = np.diff(edges)
    span = float(span)
    if not (np.isfinite(span) and span > 0):
        return 0.0
    rel = widths / span
    # Count widths that are extremely tiny vs span
    return float(np.mean(rel <= policy.microbin_rel_warn))


def _scan_parquets_for_clip_rates(
    *,
    spec: Dict[str, Any],
    quality_scores_by_file_csv: Path,
    policy: BinsHealthPolicy,
) -> Dict[str, Any]:
    qdf = _read_csv_usecols(quality_scores_by_file_csv, usecols=["parquet_path"])
    if "parquet_path" not in qdf.columns:
        raise BinsHealthError("quality_scores_by_file.csv missing 'parquet_path' column.")

    paths = [Path(str(p)) for p in qdf["parquet_path"].dropna().tolist()]
    paths = _dedupe_paths_preserve_order(paths)
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise BinsHealthError("No existing parquet paths found to scan for clipping.")

    rng = np.random.default_rng(int(policy.seed))
    sample_n = min(int(policy.parquet_sample_n), len(paths))
    if sample_n <= 0:
        return {"scanned": False, "reason": "parquet_sample_n <= 0"}

    sampled_paths = [paths[i] for i in rng.choice(len(paths), size=sample_n, replace=False)]

    # Prepare supports per laser
    ch_map = {c["laser"]: c for c in spec["channels"]}
    lasers = list(ch_map.keys())

    supports = {}
    for laser in lasers:
        ch = ch_map[laser]
        amp_edges = np.asarray(ch["amplitude_edges"], dtype=float)
        dx_edges = np.asarray(ch["increment_edges"], dtype=float)
        coup_edges = np.asarray(ch.get("coupling_edges", amp_edges), dtype=float)
        supports[laser] = {
            "amp_lo": float(amp_edges[0]),
            "amp_hi": float(amp_edges[-1]),
            "dx_lo": float(dx_edges[0]),
            "dx_hi": float(dx_edges[-1]),
            "coup_lo": float(coup_edges[0]),
            "coup_hi": float(coup_edges[-1]),
        }

    # Accumulators
    acc = {
        laser: {
            "n_amp": 0, "n_amp_oob": 0, "n_amp_nan": 0,
            "n_dx": 0, "n_dx_oob": 0,
            "n_coup": 0, "n_coup_oob": 0,
        } for laser in lasers
    }

    for p in sampled_paths:
        # read only laser columns
        cols = lasers
        df = _read_parquet_cols(p, cols=cols, engine=policy.parquet_engine)
        if df.empty:
            continue

        # cap rows to keep RAM stable
        if len(df) > int(policy.per_parquet_row_cap):
            df = df.iloc[: int(policy.per_parquet_row_cap), :].copy()

        arr = df.to_numpy(dtype=float, copy=False)  # shape (N, d)
        N = int(arr.shape[0])
        if N <= 0:
            continue

        for j, laser in enumerate(lasers):
            x = arr[:, j]
            s = supports[laser]

            finite = np.isfinite(x)
            n_f = int(np.sum(finite))
            n_nan = int(N - n_f)

            acc[laser]["n_amp"] += int(n_f)
            acc[laser]["n_amp_nan"] += int(n_nan)

            if n_f > 0:
                xf = x[finite]
                oob = np.sum((xf < s["amp_lo"]) | (xf > s["amp_hi"]))
                acc[laser]["n_amp_oob"] += int(oob)

                # coupling uses same variable; check oob vs coupling support (usually same)
                oob_c = np.sum((xf < s["coup_lo"]) | (xf > s["coup_hi"]))
                acc[laser]["n_coup"] += int(n_f)
                acc[laser]["n_coup_oob"] += int(oob_c)

                # increments
                if xf.size >= 2:
                    dx = np.diff(xf)
                    dx_finite = np.isfinite(dx)
                    dxf = dx[dx_finite]
                    acc[laser]["n_dx"] += int(dxf.size)
                    if dxf.size > 0:
                        oob_dx = np.sum((dxf < s["dx_lo"]) | (dxf > s["dx_hi"]))
                        acc[laser]["n_dx_oob"] += int(oob_dx)

    # finalize rates
    per_laser = {}
    for laser in lasers:
        a = acc[laser]
        n_amp = max(1, a["n_amp"])
        n_dx = max(1, a["n_dx"])
        n_c = max(1, a["n_coup"])
        per_laser[laser] = {
            "clip_rate_amp": float(a["n_amp_oob"] / n_amp),
            "nan_rate_amp": float(a["n_amp_nan"] / max(1, a["n_amp"] + a["n_amp_nan"])),
            "clip_rate_dx": float(a["n_dx_oob"] / n_dx),
            "clip_rate_coupling": float(a["n_coup_oob"] / n_c),
            "n_amp_finite": int(a["n_amp"]),
            "n_dx_finite": int(a["n_dx"]),
            "n_coup_finite": int(a["n_coup"]),
        }

    return {
        "scanned": True,
        "quality_scores_by_file_csv": str(quality_scores_by_file_csv),
        "n_parquets_total": int(len(paths)),
        "n_parquets_sampled": int(sample_n),
        "sampled_paths": [str(p) for p in sampled_paths],
        "per_laser": per_laser,
    }


def _merge_scan_stats_into_rows(rows: List[Dict[str, Any]], scan: Dict[str, Any]) -> None:
    if not scan or not scan.get("scanned", False):
        return
    per = scan.get("per_laser", {}) or {}
    for r in rows:
        laser = r["laser"]
        if laser in per:
            r.update(per[laser])


def _read_csv_usecols(path: Path, usecols: Sequence[str]) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, usecols=list(usecols), encoding=enc)
            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    raise BinsHealthError(f"Failed to read CSV: {path} ({type(last_err).__name__}: {last_err})")


def _read_parquet_cols(path: Path, cols: Sequence[str], engine: str = "auto") -> pd.DataFrame:
    try:
        if engine == "auto":
            return pd.read_parquet(path, columns=list(cols))
        return pd.read_parquet(path, columns=list(cols), engine=engine)
    except TypeError:
        # backend doesn't support columns kwarg
        df = pd.read_parquet(path) if engine == "auto" else pd.read_parquet(path, engine=engine)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise BinsHealthError(f"Parquet {path} missing columns: {missing}")
        return df[list(cols)]
    except Exception as e:
        raise BinsHealthError(f"Failed reading parquet: {path} ({type(e).__name__}: {e})") from e


def _dedupe_paths_preserve_order(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        out.append(Path(s))
    return out


# =============================================================================
# CLI (optional)
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate global bins health for Level 4.")
    ap.add_argument("--bins-spec", required=True, help="Path to bins_spec.json")
    ap.add_argument("--queue", default=None, help="Path to quality_scores_by_file.csv (optional, enables clipping scan)")
    ap.add_argument("--out-dir", default=None, help="Output directory for report files")
    ap.add_argument("--no-scan", action="store_true", help="Disable parquet scanning (contract checks only)")
    ap.add_argument("--sample-n", type=int, default=30, help="Number of parquets to scan")
    ap.add_argument("--row-cap", type=int, default=200_000, help="Row cap per parquet")
    args = ap.parse_args(list(argv) if argv is not None else None)

    pol = BinsHealthPolicy(
        scan_parquets=not args.no_scan and args.queue is not None,
        parquet_sample_n=int(args.sample_n),
        per_parquet_row_cap=int(args.row_cap),
    )

    report = evaluate_bins_health(
        bins_spec_json=Path(args.bins_spec),
        quality_scores_by_file_csv=Path(args.queue) if args.queue else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        policy=pol,
    )
    print_bins_health_report(report)
    return 0 if report.get("status") != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
