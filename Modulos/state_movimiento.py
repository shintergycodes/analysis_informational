# state_movimiento.py
from __future__ import annotations

"""
State: MOVIMIENTO (Increments)

Formatter / presenter module for the MOVIMIENTO state in the current pipeline.

It does NOT recompute PMFs from raw Parquet. Instead it consumes artifacts
produced by `informational_states.py`:

- Reports/Level4_Informational/States/states_summary.csv
- Reports/Level4_Informational/States/pmf_long.parquet  (or pmf_long.csv)

and produces organized, human-friendly outputs:

- Reports/Level4_Informational/Movimiento/movimiento_summary.csv
- Reports/Level4_Informational/Movimiento/movimiento_pmf_long.csv (optional)
- Reports/Level4_Informational/Movimiento/by_jkey/...             (optional)

Blind/declared support:
- If states_summary contains bins_mode == "blind", color is redacted by default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import json
import numpy as np
import pandas as pd


# =============================================================================
# Exceptions
# =============================================================================

class MovimientoStateError(RuntimeError):
    """Raised when MOVIMIENTO state presentation cannot be produced."""


# =============================================================================
# Policy / Artifacts
# =============================================================================

@dataclass(frozen=True)
class MovimientoPolicy:
    """
    Policy for generating MOVIMIENTO outputs from informational states artifacts.
    """
    # Grouping / organization
    split_by_jkey: bool = True
    jkey_fields: Tuple[str, ...] = ("fecha", "lab")
    jkey_sep: str = "_"

    # Output selection
    write_pmf_long: bool = False  # if True, reads pmf_long.* and writes movement-only copy

    # Blind-mode safety
    redact_labels_if_blind: bool = True
    redacted_value: str = "REDACTED"

    # Summary statistics
    std_ddof: int = 0  # 0 = population std (recommended for "state" dispersion)

    # IO robustness
    csv_encoding: str = "utf-8"
    allow_csv_fallback_for_parquet: bool = True  # if pmf_long.parquet can't be read, try pmf_long.csv

    # Sorting
    sort_by: Tuple[str, ...] = ("fecha", "lab", "mid")

    # Feature aliases for pmf_long filtering
    pmf_feature_aliases: Tuple[str, ...] = ("movement", "increments", "increment", "inc", "mov")


@dataclass(frozen=True)
class MovimientoArtifacts:
    out_dir: Path
    movimiento_summary_csv: Path
    movimiento_pmf_long_csv: Optional[Path]
    by_jkey_dir: Optional[Path]
    run_meta_json: Path


# =============================================================================
# Public API
# =============================================================================

def run_state_movimiento(
    *,
    states_reports_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    policy: MovimientoPolicy = MovimientoPolicy(),
    verbose: bool = True,
) -> MovimientoArtifacts:
    """
    Build MOVIMIENTO (Increments) presentation artifacts.

    Parameters
    ----------
    states_reports_dir:
        Path to Reports/Level4_Informational/States
    out_dir:
        Output directory. If None, defaults to sibling folder:
        Reports/Level4_Informational/Movimiento
    policy:
        MovimientoPolicy controlling grouping and outputs.

    Returns
    -------
    MovimientoArtifacts with output paths.
    """
    states_reports_dir = Path(states_reports_dir)
    _require_dir(states_reports_dir, "states_reports_dir")

    states_summary_path = states_reports_dir / "states_summary.csv"
    _require_file(states_summary_path, "states_summary.csv")

    if out_dir is None:
        # Expected layout: Reports/Level4_Informational/States -> Reports/Level4_Informational/Movimiento
        out_dir = states_reports_dir.parent / "Movimiento"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_jkey_dir = out_dir / "by_jkey" if policy.split_by_jkey else None
    if by_jkey_dir is not None:
        by_jkey_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 72)
        print("[Level4] State MOVIMIENTO — building organized outputs from informational states artifacts")
        print("=" * 72)
        print(f"[IN ] States dir : {states_reports_dir}")
        print(f"[IN ] states_summary.csv : {states_summary_path}")
        print(f"[OUT] Movimiento dir : {out_dir}")
        if by_jkey_dir is not None:
            print(f"[OUT] by_jkey dir : {by_jkey_dir}")

    # Load Level 2 summary and build MOVIMIENTO summary
    ss = _read_csv_robust(states_summary_path)
    ss = _normalize_columns(ss)
    _validate_states_summary_contract(ss)

    # Optional redact if blind
    bins_mode = _infer_bins_mode(ss)
    if policy.redact_labels_if_blind and bins_mode == "blind":
        ss = _redact_labels(ss, redacted_value=policy.redacted_value)

    movimiento_summary = _build_movimiento_summary(ss, policy=policy)

    # Write summary
    movimiento_summary_csv = out_dir / "movimiento_summary.csv"
    movimiento_summary.to_csv(movimiento_summary_csv, index=False, encoding=policy.csv_encoding)

    if verbose:
        print(f"[OK] wrote: {movimiento_summary_csv}  (rows={len(movimiento_summary)})")

    # Split by jkey (summary)
    if by_jkey_dir is not None:
        _write_by_jkey_csv(
            df=movimiento_summary,
            by_jkey_dir=by_jkey_dir,
            prefix="Movimiento_Resumen",
            jkey_col="_jkey",
            encoding=policy.csv_encoding,
            verbose=verbose,
        )

    # Optional PMF long copy (movement-only)
    movimiento_pmf_long_csv: Optional[Path] = None
    if policy.write_pmf_long:
        pmf_df = _load_pmf_long(states_reports_dir, allow_csv_fallback=policy.allow_csv_fallback_for_parquet)
        pmf_df = _normalize_columns(pmf_df)

        if "feature" not in pmf_df.columns:
            raise MovimientoStateError("pmf_long is missing required column: feature")

        feat = pmf_df["feature"].astype(str).str.strip().str.lower()
        aliases = {a.lower() for a in policy.pmf_feature_aliases}
        pmf_mov = pmf_df[feat.isin(aliases)].copy()

        if pmf_mov.empty:
            raise MovimientoStateError(
                "pmf_long has no rows for MOVIMIENTO. "
                f"Expected feature in {sorted(aliases)}"
            )

        # Optional redact if blind (pmf_long includes metadata written by Level2)
        if policy.redact_labels_if_blind and bins_mode == "blind":
            pmf_mov = _redact_labels(pmf_mov, redacted_value=policy.redacted_value)

        # Add jkey if needed for splitting
        pmf_mov["_jkey"] = _make_jkey_series(pmf_mov, policy.jkey_fields, sep=policy.jkey_sep)

        movimiento_pmf_long_csv = out_dir / "movimiento_pmf_long.csv"
        pmf_mov.to_csv(movimiento_pmf_long_csv, index=False, encoding=policy.csv_encoding)

        if verbose:
            print(f"[OK] wrote: {movimiento_pmf_long_csv}  (rows={len(pmf_mov)})")

        if by_jkey_dir is not None:
            _write_by_jkey_csv(
                df=pmf_mov,
                by_jkey_dir=by_jkey_dir,
                prefix="Movimiento_PMF",
                jkey_col="_jkey",
                encoding=policy.csv_encoding,
                verbose=verbose,
            )

    # Run meta
    run_meta = {
        "module": "state_movimiento.py",
        "mode": bins_mode,
        "policy": _policy_to_jsonable(policy),
        "inputs": {
            "states_reports_dir": str(states_reports_dir),
            "states_summary_csv": str(states_summary_path),
            "pmf_long_present": _pmf_long_presence(states_reports_dir),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "movimiento_summary_csv": str(movimiento_summary_csv),
            "movimiento_pmf_long_csv": str(movimiento_pmf_long_csv) if movimiento_pmf_long_csv else None,
            "by_jkey_dir": str(by_jkey_dir) if by_jkey_dir else None,
        },
        "counts": {
            "summary_rows": int(len(movimiento_summary)),
            "unique_mids": int(movimiento_summary["mid"].nunique()) if "mid" in movimiento_summary.columns else None,
            "unique_lasers": int(_count_unique_lasers(ss)),
        },
    }
    run_meta_json = out_dir / "movimiento_run_meta.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding=policy.csv_encoding)

    if verbose:
        print(f"[OK] wrote: {run_meta_json}")

    return MovimientoArtifacts(
        out_dir=out_dir,
        movimiento_summary_csv=movimiento_summary_csv,
        movimiento_pmf_long_csv=movimiento_pmf_long_csv,
        by_jkey_dir=by_jkey_dir,
        run_meta_json=run_meta_json,
    )


# =============================================================================
# Core builders
# =============================================================================

def _build_movimiento_summary(states_summary: pd.DataFrame, *, policy: MovimientoPolicy) -> pd.DataFrame:
    """
    Build the main MOVIMIENTO summary table.

    Input is states_summary.csv from informational states (long by laser).
    Output is one row per mid, including:
      - per-laser Hinc_<laser> columns (wide) from H_mov
      - mu_mov, sigma_mov, norm2_mov
      - n_valid_channels, n_total_channels
      - metadata (first row per mid)

    Optional:
      - if N_mov is present in states_summary, also pivots N_used_<laser>.
    """
    df = states_summary.copy()

    # Build jkey
    df["_jkey"] = _make_jkey_series(df, policy.jkey_fields, sep=policy.jkey_sep)

    # Ensure required columns exist
    for col in ("mid", "laser", "H_mov"):
        if col not in df.columns:
            raise MovimientoStateError(f"states_summary missing required column: {col}")

    # Metadata columns we carry from informational states artifacts (best-effort)
    meta_cols = [
        c for c in (
            "mid", "parquet_path", "fecha", "lab",
            "archivo", "color",
            "N", "d", "bins_mode", "reference_group", "bins_spec_path",
        )
        if c in df.columns
    ]
    if "mid" not in meta_cols:
        meta_cols = ["mid"] + meta_cols
    meta = df[meta_cols + ["_jkey"]].groupby("mid", as_index=False).first()

    # Pivot entropies to wide vector (Hinc_<laser>)
    pivot_H = (
        df.pivot_table(index="mid", columns="laser", values="H_mov", aggfunc="first")
        .reset_index()
    )
    rename_map = {}
    for c in pivot_H.columns:
        if c != "mid":
            rename_map[c] = f"Hinc_{c}"
    pivot_H = pivot_H.rename(columns=rename_map)

    h_cols = [c for c in pivot_H.columns if c.startswith("Hinc_")]
    Hmat = pivot_H[h_cols].to_numpy(dtype=float, copy=False) if h_cols else np.zeros((len(pivot_H), 0), dtype=float)

    n_total = Hmat.shape[1]
    n_valid = np.sum(np.isfinite(Hmat), axis=1)

    mu = np.nanmean(Hmat, axis=1) if n_total > 0 else np.full(len(pivot_H), np.nan)
    sigma = np.nanstd(Hmat, axis=1, ddof=int(policy.std_ddof)) if n_total > 1 else np.full(len(pivot_H), np.nan)
    norm2 = np.sqrt(np.nansum(Hmat * Hmat, axis=1)) if n_total > 0 else np.full(len(pivot_H), np.nan)

    pivot_H["n_total_channels"] = int(n_total)
    pivot_H["n_valid_channels"] = n_valid.astype(int)
    pivot_H["mu_mov"] = mu
    pivot_H["sigma_mov"] = sigma
    pivot_H["norm2_mov"] = norm2

    out = meta.merge(pivot_H, on="mid", how="left")

    # Optional: pivot N_mov (if present) into N_used_<laser>
    if "N_mov" in df.columns:
        pivot_N = (
            df.pivot_table(index="mid", columns="laser", values="N_mov", aggfunc="first")
            .reset_index()
        )
        renameN = {c: (f"N_used_{c}" if c != "mid" else c) for c in pivot_N.columns}
        pivot_N = pivot_N.rename(columns=renameN)
        out = out.merge(pivot_N, on="mid", how="left")

    # Reorder: meta, aggregates, vector, optional N_used
    agg_cols = ["n_total_channels", "n_valid_channels", "mu_mov", "sigma_mov", "norm2_mov"]
    n_cols = [c for c in out.columns if c.startswith("N_used_")]

    ordered = [c for c in meta.columns if c != "_jkey"] + ["_jkey"]
    ordered += [c for c in agg_cols if c in out.columns]
    ordered += h_cols
    ordered += n_cols
    ordered = [c for c in ordered if c in out.columns] + [c for c in out.columns if c not in ordered]
    out = out[ordered]

    out = _safe_sort(out, policy.sort_by)
    return out


# =============================================================================
# PMF loading
# =============================================================================

def _load_pmf_long(states_reports_dir: Path, *, allow_csv_fallback: bool) -> pd.DataFrame:
    """
    Load pmf_long from informational states artifacts, preferring parquet and falling back to csv.
    """
    pq_path = states_reports_dir / "pmf_long.parquet"
    csv_path = states_reports_dir / "pmf_long.csv"

    if pq_path.exists():
        try:
            return pd.read_parquet(pq_path)
        except Exception as e:
            if not allow_csv_fallback:
                raise MovimientoStateError(f"Failed to read pmf_long.parquet and csv fallback disabled: {e}") from e
            if csv_path.exists():
                return _read_csv_robust(csv_path)
            raise MovimientoStateError(f"Failed to read pmf_long.parquet and pmf_long.csv not found: {e}") from e

    if csv_path.exists():
        return _read_csv_robust(csv_path)

    raise MovimientoStateError("Neither pmf_long.parquet nor pmf_long.csv exists in states reports dir.")


def _pmf_long_presence(states_reports_dir: Path) -> Dict[str, bool]:
    return {
        "pmf_long.parquet": (states_reports_dir / "pmf_long.parquet").exists(),
        "pmf_long.csv": (states_reports_dir / "pmf_long.csv").exists(),
    }


# =============================================================================
# Validation
# =============================================================================

def _validate_states_summary_contract(df: pd.DataFrame) -> None:
    required = ("mid", "laser", "H_mov")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MovimientoStateError(f"states_summary.csv missing required columns: {missing}")

    if df["laser"].isna().any():
        raise MovimientoStateError("states_summary.csv has missing laser values.")

    df["H_mov"] = pd.to_numeric(df["H_mov"], errors="coerce")
    if "N_mov" in df.columns:
        df["N_mov"] = pd.to_numeric(df["N_mov"], errors="coerce")


def _infer_bins_mode(states_summary: pd.DataFrame) -> str:
    if "bins_mode" in states_summary.columns:
        s = states_summary["bins_mode"].dropna()
        if not s.empty:
            return str(s.iloc[0]).strip().lower()
    return ""


def _count_unique_lasers(states_summary: pd.DataFrame) -> int:
    if "laser" not in states_summary.columns:
        return 0
    return int(states_summary["laser"].astype(str).nunique())


def _redact_labels(df: pd.DataFrame, *, redacted_value: str) -> pd.DataFrame:
    out = df.copy()
    for col in ("color",):
        if col in out.columns:
            out[col] = redacted_value
    return out


# =============================================================================
# jkey helpers
# =============================================================================

def _make_jkey_series(df: pd.DataFrame, fields: Sequence[str], *, sep: str) -> pd.Series:
    parts = []
    for f in fields:
        if f in df.columns:
            parts.append(df[f].astype(str).fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    jkey = parts[0]
    for p in parts[1:]:
        jkey = jkey + sep + p
    return jkey.str.replace(r"\s+", "", regex=True)


def _write_by_jkey_csv(
    *,
    df: pd.DataFrame,
    by_jkey_dir: Path,
    prefix: str,
    jkey_col: str,
    encoding: str,
    verbose: bool,
) -> None:
    if jkey_col not in df.columns:
        raise MovimientoStateError(f"Cannot split by jkey: missing column {jkey_col}")

    for jkey, g in df.groupby(jkey_col):
        safe = _sanitize_filename(str(jkey)) or "UNK"
        out_path = by_jkey_dir / f"{prefix}_{safe}.csv"
        g.to_csv(out_path, index=False, encoding=encoding)
        if verbose:
            print(f"[OK] wrote: {out_path}  (rows={len(g)})")


def _sanitize_filename(s: str) -> str:
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        s = s.replace(ch, "")
    return s.strip()


# =============================================================================
# Misc IO helpers
# =============================================================================

def _require_dir(path: Path, label: str) -> None:
    if not Path(path).exists() or not Path(path).is_dir():
        raise MovimientoStateError(f"{label} must be an existing directory: {path}")


def _require_file(path: Path, label: str) -> None:
    if not Path(path).exists() or not Path(path).is_file():
        raise MovimientoStateError(f"{label} not found: {path}")


def _read_csv_robust(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise MovimientoStateError(f"Failed to read CSV: {path} ({type(last_err).__name__}: {last_err})")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]
    return out


def _safe_sort(df: pd.DataFrame, sort_by: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in sort_by if c in df.columns]
    if not cols:
        return df
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def _policy_to_jsonable(policy: MovimientoPolicy) -> Dict[str, Any]:
    return {
        "split_by_jkey": bool(policy.split_by_jkey),
        "jkey_fields": list(policy.jkey_fields),
        "jkey_sep": str(policy.jkey_sep),
        "write_pmf_long": bool(policy.write_pmf_long),
        "redact_labels_if_blind": bool(policy.redact_labels_if_blind),
        "redacted_value": str(policy.redacted_value),
        "std_ddof": int(policy.std_ddof),
        "csv_encoding": str(policy.csv_encoding),
        "allow_csv_fallback_for_parquet": bool(policy.allow_csv_fallback_for_parquet),
        "sort_by": list(policy.sort_by),
        "pmf_feature_aliases": list(policy.pmf_feature_aliases),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args(argv: Optional[Sequence[str]] = None):
    import argparse

    p = argparse.ArgumentParser(description="State MOVIMIENTO (Increments) — formatter for informational states artifacts.")
    p.add_argument("--states-reports-dir", required=True, help="Path to Reports/Level4_Informational/States")
    p.add_argument("--out-dir", default=None, help="Output directory (default: Reports/Level4_Informational/Movimiento)")
    p.add_argument("--no-split-by-jkey", action="store_true", help="Disable by_jkey outputs")
    p.add_argument("--write-pmf-long", action="store_true", help="Write movement-only PMF long CSV")
    p.add_argument("--jkey-fields", default="fecha,lab", help="Comma-separated jkey fields")
    p.add_argument("--no-redact-blind", action="store_true", help="Do not redact color in blind mode")
    p.add_argument("--verbose", action="store_true", help="Verbose console logs")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    policy = MovimientoPolicy(
        split_by_jkey=not bool(args.no_split_by_jkey),
        write_pmf_long=bool(args.write_pmf_long),
        jkey_fields=tuple([x.strip() for x in str(args.jkey_fields).split(",") if x.strip()]),
        redact_labels_if_blind=not bool(args.no_redact_blind),
    )

    try:
        run_state_movimiento(
            states_reports_dir=args.states_reports_dir,
            out_dir=args.out_dir,
            policy=policy,
            verbose=bool(args.verbose),
        )
        return 0
    except Exception as e:
        print("\n" + "!" * 72)
        print("[FAIL] state_movimiento.py")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
