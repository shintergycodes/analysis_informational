# state_fourier.py
from __future__ import annotations

"""
State: FOURIER (Modal spectral entropy) — PSD-cache-first

This module computes a Fourier-based informational state per measurement and per laser channel.
It is designed for later inference about "changes between measurements", so it prioritizes:

- Strict comparability across measurements (same frequency grid and window per laser),
- Robustness (uses averaged PSD cache produced by state_energia.py),
- Determinism and low resource usage (streaming; per-measurement processing).

Primary input is the PSD cache written by state_energia.py:
  Reports/Level4_Informational/Energia/psd_cache/psd_<MID_SANITIZED>_<LASER_SANITIZED>.npz
  
Raw-FFT fallback exists but is disabled by default (not recommended for inference).

All code and console messages are in English.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import csv
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# Exceptions
# =============================================================================

class FourierStateError(RuntimeError):
    """Raised when the FOURIER state computation cannot be produced."""


# =============================================================================
# Policy / Artifacts
# =============================================================================

@dataclass(frozen=True)
class FourierPolicy:
    # Organization
    split_by_jkey: bool = True
    jkey_fields: Tuple[str, ...] = ("fecha", "lab")
    jkey_sep: str = "_"
    
    # Blind-mode safety
    redact_labels_if_blind: bool = True
    redacted_value: str = "REDACTED"
    drop_archivo: bool = True
    drop_parquet_path: bool = True

    # PSD-cache-first behavior
    psd_first_required: bool = True
    allow_raw_fallback: bool = False  # strongly recommended False for inference comparability
    psd_cache_dir_override: Optional[Union[str, Path]] = None
    energia_run_meta_override: Optional[Union[str, Path]] = None

    # Frequency window (units match PSD freqs; typically cycles/sample or Hz)
    fmin: float = 0.0
    fmax: Optional[float] = None  # if None, default to max(freq_edges) per laser from bins_spec

    # Grid consistency
    enforce_freq_grid_consistency: bool = True
    freq_grid_rtol: float = 0.0
    freq_grid_atol: float = 1e-12

    # PMF / entropy
    exclude_dc: bool = True  # forced True
    alpha: float = 0.0  # pseudocount in energy space (>=0)
    clamp_negative_psd_to_zero: bool = True

    # Outputs
    write_pmf_long: bool = False  # can be huge; default OFF
    pmf_long_max_modes: Optional[int] = None  # if set, keep only top-energy modes in long output
    csv_encoding: str = "utf-8"
    sort_by: Tuple[str, ...] = ("fecha", "lab", "mid")

    # Stats
    std_ddof: int = 0  # 0 = population std


@dataclass(frozen=True)
class FourierArtifacts:
    out_dir: Path
    fourier_summary_csv: Path
    fourier_pmf_long_csv: Optional[Path]
    by_jkey_dir: Optional[Path]
    run_meta_json: Path


# =============================================================================
# Public API
# =============================================================================



def run_state_fourier(
    *,
    informational_queue_csv: Union[str, Path],
    bins_spec_json: Union[str, Path],
    out_dir: Union[str, Path],
    lasers: Sequence[str],
    mode: str = "",
    policy: FourierPolicy = FourierPolicy(),
    verbose: bool = True,
) -> FourierArtifacts:
    """
    Core runner: computes FOURIER state from queue + bins_spec using PSD cache (preferred).

    Semantics:
    - ENERGIA = spectral organization by frequency bands
    - FOURIER = fine modal spectral entropy on the PSD grid, without rebinning
    """
    qpath = Path(informational_queue_csv)
    bpath = Path(bins_spec_json)
    out_dir = Path(out_dir)

    _require_file(qpath, "informational_queue_csv")
    _require_file(bpath, "bins_spec_json")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_jkey_dir = out_dir / "by_jkey" if policy.split_by_jkey else None
    if by_jkey_dir is not None:
        by_jkey_dir.mkdir(parents=True, exist_ok=True)

    # Long output (streaming CSV)
    fourier_pmf_long_csv: Optional[Path] = None
    pmf_writer = None
    pmf_fh = None
    if policy.write_pmf_long:
        fourier_pmf_long_csv = out_dir / "fourier_pmf_long.csv"
        pmf_fh = open(fourier_pmf_long_csv, "w", newline="", encoding=policy.csv_encoding)
        pmf_writer = csv.DictWriter(
            pmf_fh,
            fieldnames=["mid", "jkey", "laser", "mode_index", "freq", "p", "E"],
        )
        pmf_writer.writeheader()

    # Load specs
    bins_spec = _read_json(bpath)
    bins_hash = _sha256_bytes(bpath.read_bytes())
    spectral_map = _extract_spectral_specs(bins_spec, lasers)

    # Resolve fmax defaults per laser from bins_spec
    fmax_default_by_laser = {
        laser: float(np.max(np.asarray(spectral_map[laser]["freq_edges"], dtype=float)))
        for laser in lasers
    }

    # Queue
    queue = _read_queue(qpath)
    queue = _normalize_queue_columns(queue)
    _require_columns(queue, ["mid"], "informational_queue.csv")
    queue["_jkey"] = _make_jkey_series(queue, policy.jkey_fields, sep=policy.jkey_sep)

    mode = (mode or "").strip().lower()

    # Resolve PSD cache directory
    psd_cache_dir, energy_meta_path = _resolve_psd_cache_dir(
        policy=policy,
        out_dir=out_dir,
    )

    if verbose:
        print("\n" + "=" * 72)
        print("[Level4] Building FOURIER state (modal spectral entropy, PSD-cache-first)")
        print("=" * 72)
        print(f"[IN ] queue     : {qpath}")
        print(f"[IN ] bins_spec : {bpath}  (sha256={bins_hash[:12]}...)")
        print(f"[IN ] psd_cache : {psd_cache_dir}")
        if energy_meta_path:
            print(f"[IN ] energia_run_meta.json : {energy_meta_path}")
        print(f"[OUT] dir       : {out_dir}")
        print(f"[MODE] {mode or 'unknown'}")
        print(f"[LASERS] {list(lasers)}")
        if fourier_pmf_long_csv:
            print(f"[OUT] fourier_pmf_long.csv : {fourier_pmf_long_csv}")

    # Enforce invariant
    if not policy.exclude_dc:
        raise FourierStateError("exclude_dc must be True for this state (DC is always excluded).")

    # Reference frequency grids (for strict comparability)
    freq_ref_by_laser: Dict[str, Optional[np.ndarray]] = {l: None for l in lasers}

    # Iterate measurement-by-measurement
    summary_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_partial = 0
    n_fail = 0
    n_cache_used = 0
    n_cache_missing = 0
    n_raw_fallback_used = 0

    for idx, row in queue.iterrows():
        mid = _safe_str(row.get("mid")) or f"row{idx}"
        jkey = _safe_str(row.get("_jkey"))

        meta = _extract_meta(row, keep_archivo=not policy.drop_archivo)
        if policy.drop_parquet_path:
            meta.pop("parquet_path", None)

        if policy.redact_labels_if_blind and mode == "blind":
            meta = _redact_meta(meta, redacted_value=policy.redacted_value)

        H_by_laser: Dict[str, float] = {}
        n_modes_by_laser: Dict[str, int] = {}
        ok_lasers = 0

        for laser in lasers:
            H = np.nan
            n_modes = 0

            cache_path = _psd_cache_path(psd_cache_dir, mid=mid, laser=laser)
            if cache_path.exists():
                n_cache_used += 1
                freqs, psd, _cache_meta = _load_psd_cache_npz(cache_path)

                # Grid consistency check (optional but recommended)
                if policy.enforce_freq_grid_consistency:
                    ref = freq_ref_by_laser.get(laser)
                    if ref is None:
                        freq_ref_by_laser[laser] = freqs.copy()
                    else:
                        if ref.shape != freqs.shape or not np.allclose(
                            ref, freqs, rtol=policy.freq_grid_rtol, atol=policy.freq_grid_atol
                        ):
                            raise FourierStateError(
                                f"Frequency grid mismatch for laser={laser}. "
                                f"Ensure state_energia used consistent spectral settings (n_fft, fs)."
                            )

                H, pmf_rows, n_modes = _fourier_entropy_from_psd(
                    freqs=freqs,
                    psd=psd,
                    fmin=float(policy.fmin),
                    fmax=float(policy.fmax) if policy.fmax is not None else fmax_default_by_laser[laser],
                    exclude_dc=True,
                    alpha=float(policy.alpha),
                    clamp_negative_to_zero=bool(policy.clamp_negative_psd_to_zero),
                    pmf_long=bool(policy.write_pmf_long),
                    pmf_long_max_modes=policy.pmf_long_max_modes,
                )

                if pmf_writer is not None and pmf_rows is not None:
                    for r in pmf_rows:
                        pmf_writer.writerow(
                            {
                                "mid": mid,
                                "jkey": jkey,
                                "laser": laser,
                                "mode_index": int(r["mode_index"]),
                                "freq": float(r["freq"]),
                                "p": float(r["p"]),
                                "E": float(r["E"]),
                            }
                        )

            else:
                n_cache_missing += 1
                if policy.allow_raw_fallback:
                    parquet_path = Path(_safe_str(row.get("parquet_path")))
                    if parquet_path.exists():
                        try:
                            df = pd.read_parquet(parquet_path, columns=[laser])
                            x = df[laser].to_numpy(dtype=float, copy=False)
                            n_fft = int(spectral_map[laser]["n_fft"])
                            fs = float(spectral_map[laser]["fs"])
                            freqs, psd = _raw_rfft_psd(x=x, n_fft=n_fft, fs=fs)
                            H, pmf_rows, n_modes = _fourier_entropy_from_psd(
                                freqs=freqs,
                                psd=psd,
                                fmin=float(policy.fmin),
                                fmax=float(policy.fmax) if policy.fmax is not None else fmax_default_by_laser[laser],
                                exclude_dc=True,
                                alpha=float(policy.alpha),
                                clamp_negative_to_zero=bool(policy.clamp_negative_psd_to_zero),
                                pmf_long=bool(policy.write_pmf_long),
                                pmf_long_max_modes=policy.pmf_long_max_modes,
                            )
                            n_raw_fallback_used += 1

                            if pmf_writer is not None and pmf_rows is not None:
                                for r in pmf_rows:
                                    pmf_writer.writerow(
                                        {
                                            "mid": mid,
                                            "jkey": jkey,
                                            "laser": laser,
                                            "mode_index": int(r["mode_index"]),
                                            "freq": float(r["freq"]),
                                            "p": float(r["p"]),
                                            "E": float(r["E"]),
                                        }
                                    )
                        except Exception:
                            H = np.nan
                            n_modes = 0

            H_by_laser[laser] = float(H) if np.isfinite(H) else np.nan
            n_modes_by_laser[laser] = int(n_modes)

            if np.isfinite(H_by_laser[laser]):
                ok_lasers += 1

        # Determine status
        if ok_lasers == len(lasers):
            status = "OK"
            n_ok += 1
        elif ok_lasers > 0:
            status = "PARTIAL"
            n_partial += 1
        else:
            status = "FAIL"
            n_fail += 1

        # Aggregate across lasers
        H_valid = np.array([H_by_laser[l] for l in lasers if np.isfinite(H_by_laser[l])], dtype=float)
        mu = float(np.nanmean(H_valid)) if H_valid.size > 0 else np.nan
        sigma = float(np.nanstd(H_valid, ddof=int(policy.std_ddof))) if H_valid.size > 1 else np.nan
        norm2 = float(np.sqrt(np.nansum(H_valid * H_valid))) if H_valid.size > 0 else np.nan

        out_row: Dict[str, Any] = {}
        out_row.update(meta)
        out_row["mid"] = mid
        out_row["jkey"] = jkey
        out_row["status"] = status
        out_row["n_valid_channels"] = int(ok_lasers)
        out_row["mu_fourier"] = mu
        out_row["sigma_fourier"] = sigma
        out_row["norm2_fourier"] = norm2

        for laser in lasers:
            out_row[f"Hfourier_{laser}"] = H_by_laser[laser]
            out_row[f"n_modes_{laser}"] = n_modes_by_laser[laser]

        summary_rows.append(out_row)

    # Close long output
    if pmf_fh is not None:
        pmf_fh.close()

    # Summary output
    summary = pd.DataFrame(summary_rows)
    summary = _safe_sort(summary, policy.sort_by)
    fourier_summary_csv = out_dir / "fourier_summary.csv"
    summary.to_csv(fourier_summary_csv, index=False, encoding=policy.csv_encoding)

    # by_jkey splits (summary only)
    if by_jkey_dir is not None:
        _write_by_jkey_summary(summary, by_jkey_dir, encoding=policy.csv_encoding, verbose=verbose)

    # Meta
    run_meta = {
        "module": "state_fourier.py",
        "mode": mode or "",
        "bins_spec_sha256": bins_hash,
        "bins_spec_path": str(bpath),
        "queue_path": str(qpath),
        "psd_cache_dir": str(psd_cache_dir),
        "energia_run_meta_path": str(energy_meta_path) if energy_meta_path else None,
        "outputs": {
            "out_dir": str(out_dir),
            "fourier_summary_csv": str(fourier_summary_csv),
            "fourier_pmf_long_csv": str(fourier_pmf_long_csv) if fourier_pmf_long_csv else None,
            "by_jkey_dir": str(by_jkey_dir) if by_jkey_dir else None,
            "state_semantics": {
                "energia": "banded spectral entropy",
                "fourier": "fine modal spectral entropy on PSD grid",
            },
        },
        "counts": {
            "n_measurements": int(len(summary)),
            "n_ok": int(n_ok),
            "n_partial": int(n_partial),
            "n_fail": int(n_fail),
            "n_lasers": int(len(lasers)),
            "n_cache_used": int(n_cache_used),
            "n_cache_missing": int(n_cache_missing),
            "n_raw_fallback_used": int(n_raw_fallback_used),
        },
        "policy": _policy_to_jsonable(policy),
        "spectral_defaults": {
            "fmin": float(policy.fmin),
            "fmax_global": float(policy.fmax) if policy.fmax is not None else None,
            "fmax_default_by_laser": {l: float(fmax_default_by_laser[l]) for l in lasers},
        },
    }
    run_meta_json = out_dir / "fourier_run_meta.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding=policy.csv_encoding)

    if verbose:
        print(f"[OK] wrote: {fourier_summary_csv}  (rows={len(summary)})")
        if fourier_pmf_long_csv:
            print(f"[OK] wrote: {fourier_pmf_long_csv}")
        print(f"[OK] wrote: {run_meta_json}")

    return FourierArtifacts(
        out_dir=out_dir,
        fourier_summary_csv=fourier_summary_csv,
        fourier_pmf_long_csv=fourier_pmf_long_csv,
        by_jkey_dir=by_jkey_dir,
        run_meta_json=run_meta_json,
    )


# =============================================================================
# Core computations
# =============================================================================

def _fourier_entropy_from_psd(
    *,
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
    exclude_dc: bool,
    alpha: float,
    clamp_negative_to_zero: bool,
    pmf_long: bool,
    pmf_long_max_modes: Optional[int],
) -> Tuple[float, Optional[List[Dict[str, Any]]], int]:
    """
    Compute Shannon entropy (bits) of the modal PMF defined on PSD values.

    Returns:
        H (float), pmf_rows (optional), n_modes_used (int)
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    psd = np.asarray(psd, dtype=float).reshape(-1)

    if freqs.size != psd.size or freqs.size < 2:
        return (np.nan, None, 0)

    if not np.all(np.isfinite(freqs)) or not np.all(np.isfinite(psd)):
        return (np.nan, None, 0)
    if np.any(np.diff(freqs) <= 0):
        return (np.nan, None, 0)

    if clamp_negative_to_zero:
        psd = np.maximum(psd, 0.0)

    # Mask and original indices
    mask = np.ones_like(freqs, dtype=bool)
    if exclude_dc:
        mask &= (freqs > 0.0)
    mask &= (freqs >= float(fmin))
    mask &= (freqs <= float(fmax))

    if not np.any(mask):
        return (np.nan, None, 0)

    idx = np.flatnonzero(mask)
    freqs_sel = freqs[idx]
    E = psd[idx]

    E_sum = float(np.sum(E))
    if not np.isfinite(E_sum) or E_sum <= 0.0:
        return (np.nan, None, int(idx.size))

    K = int(E.size)
    a = float(alpha)
    if a < 0:
        a = 0.0

    if a > 0.0:
        denom = E_sum + a * K
        if denom <= 0.0:
            return (np.nan, None, K)
        p = (E + a) / denom
    else:
        p = E / E_sum

    H = _shannon_entropy_bits(p)

    pmf_rows = None
    if pmf_long:
        # Optionally keep only top-energy modes
        if pmf_long_max_modes is not None and pmf_long_max_modes > 0 and K > pmf_long_max_modes:
            # Top by energy; stable tie-break by frequency
            order = np.lexsort((freqs_sel, -E))
            keep_local = order[:pmf_long_max_modes]
            # Sort kept modes by frequency for readability
            keep_local = keep_local[np.argsort(freqs_sel[keep_local])]
        else:
            keep_local = np.arange(K, dtype=int)

        pmf_rows = [
            {
                "mode_index": int(idx[int(k)]),  # original index in PSD arrays
                "freq": float(freqs_sel[int(k)]),
                "p": float(p[int(k)]),
                "E": float(E[int(k)]),
            }
            for k in keep_local
        ]

    return (float(H) if np.isfinite(H) else np.nan, pmf_rows, int(K))


def _raw_rfft_psd(*, x: np.ndarray, n_fft: int, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Raw fallback: compute a single-block rFFT PSD on the first n_fft samples (or zero-pad).
    NOT recommended for inference if PSD cache is available.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.array([], dtype=float), np.array([], dtype=float))

    n_fft = int(n_fft)
    if n_fft <= 0:
        n_fft = int(min(4096, x.size))

    if x.size >= n_fft:
        block = x[:n_fft].copy()
    else:
        block = np.zeros(n_fft, dtype=float)
        block[: x.size] = x

    block = block - float(np.mean(block))
    X = np.fft.rfft(block, n=n_fft)
    psd = (np.abs(X) ** 2) / float(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs if fs > 0 else 1.0))
    return freqs.astype(float), psd.astype(float)


def _shannon_entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * (np.log(p) / np.log(2.0))))


# =============================================================================
# PSD cache resolution / reading
# =============================================================================



def _psd_cache_path(psd_cache_dir: Path, *, mid: str, laser: str) -> Path:
    mid_s = _sanitize_filename(mid) or "mid"
    laser_s = _sanitize_filename(laser) or "laser"
    return psd_cache_dir / f"psd_{mid_s}_{laser_s}.npz"


def _load_psd_cache_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    freqs = np.asarray(z["freqs"], dtype=float)
    psd = np.asarray(z["psd"], dtype=float)
    meta: Dict[str, Any] = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {}
    return freqs, psd, meta


# =============================================================================
# Queue / spec parsing
# =============================================================================

def _read_queue(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="latin-1")


def _normalize_queue_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]

    rename = {}
    if "laboratorio" in out.columns and "lab" not in out.columns:
        rename["laboratorio"] = "lab"
    if "measurement_id" in out.columns and "mid" not in out.columns:
        rename["measurement_id"] = "mid"
    if "path" in out.columns and "parquet_path" not in out.columns:
        rename["path"] = "parquet_path"
    if rename:
        out = out.rename(columns=rename)

    if "mid" not in out.columns:
        parts = []
        if "parquet_path" in out.columns:
            parts.append(out["parquet_path"].astype(str))
        for c in ("fecha", "lab"):
            if c in out.columns:
                parts.append(out[c].astype(str))
        if parts:
            base = parts[0]
            for p in parts[1:]:
                base = base + "|" + p
            out["mid"] = base.map(lambda s: _short_hash(s))
        else:
            out["mid"] = [f"row{i}" for i in range(len(out))]

    return out

def _resolve_psd_cache_dir(
    *,
    policy: FourierPolicy,
    out_dir: Path,
) -> Tuple[Path, Optional[Path]]:
    """
    Resolve the PSD cache directory from explicit overrides or from the sibling
    Level4 Energia folder.

    Returns (psd_cache_dir, energia_run_meta_path_or_None).
    """
    if policy.psd_cache_dir_override is not None:
        d = Path(policy.psd_cache_dir_override)
        _require_dir(d, "psd_cache_dir_override")
        return d, None

    energia_meta_path: Optional[Path] = None

    if policy.energia_run_meta_override is not None:
        energia_meta_path = Path(policy.energia_run_meta_override)
        _require_file(energia_meta_path, "energia_run_meta_override")
        meta = _read_json(energia_meta_path)
        psd_dir = meta.get("outputs", {}).get("psd_cache_dir")
        if psd_dir:
            d = Path(psd_dir)
            _require_dir(d, "psd_cache_dir from energia_run_meta")
            return d, energia_meta_path

    energia_dir = out_dir.parent / "Energia"
    candidate_meta = energia_dir / "energia_run_meta.json"
    if candidate_meta.exists():
        energia_meta_path = candidate_meta
        meta = _read_json(candidate_meta)
        psd_dir = meta.get("outputs", {}).get("psd_cache_dir")
        if psd_dir:
            d = Path(psd_dir)
            if d.exists() and d.is_dir():
                return d, energia_meta_path

    candidate_cache = energia_dir / "psd_cache"
    if candidate_cache.exists() and candidate_cache.is_dir():
        return candidate_cache, energia_meta_path

    if policy.psd_first_required:
        raise FourierStateError(
            "PSD cache directory not found. Run state_energia with write_psd_cache=True, "
            "or provide psd_cache_dir_override / energia_run_meta_override."
        )

    return candidate_cache, energia_meta_path


def _extract_meta(row: pd.Series, *, keep_archivo: bool) -> dict:
    keep = ["fecha", "lab", "color", "parquet_path"]
    if keep_archivo:
        keep.append("archivo")
    meta = {}
    for k in keep:
        if k in row.index:
            meta[k] = _safe_str(row.get(k))
    return meta

def _redact_meta(meta: dict, *, redacted_value: str) -> dict:
    out = dict(meta)
    for k in ("color", "archivo"):
        if k in out and out[k] not in ("", None):
            out[k] = redacted_value
    return out


def _extract_spectral_specs(bins_spec: dict, lasers: Sequence[str]) -> dict:
    channels = None
    if isinstance(bins_spec.get("bins"), list):
        channels = bins_spec.get("bins")
    elif isinstance(bins_spec.get("channels"), list):
        channels = bins_spec.get("channels")
    if not channels:
        raise FourierStateError("bins_spec.json must contain a list at key 'bins' or 'channels'")

    by_laser = {}
    for ch in channels:
        laser = _safe_str(ch.get("laser"))
        if not laser or laser not in lasers:
            continue

        spectral = ch.get("spectral")
        if not isinstance(spectral, dict):
            raise FourierStateError(f"bins_spec missing spectral section for laser={laser}")

        freq_edges = spectral.get("freq_edges")
        if freq_edges is None:
            raise FourierStateError(f"bins_spec spectral.freq_edges missing for laser={laser}")

        freq_edges_arr = np.asarray(freq_edges, dtype=float)
        if freq_edges_arr.size < 2 or not np.all(np.isfinite(freq_edges_arr)):
            raise FourierStateError(f"bins_spec spectral.freq_edges invalid for laser={laser}")
        if not np.all(np.diff(freq_edges_arr) > 0):
            raise FourierStateError(f"bins_spec spectral.freq_edges not strictly increasing for laser={laser}")

        n_fft = int(spectral.get("n_fft", 4096))
        fs = float(spectral.get("fs", 1.0))

        by_laser[laser] = {
            "freq_edges": freq_edges_arr,
            "n_fft": n_fft,
            "fs": fs,
            "freq_unit": str(spectral.get("freq_unit", "")),
            "method": str(spectral.get("method", "")),
        }

    missing = [l for l in lasers if l not in by_laser]
    if missing:
        raise FourierStateError(f"bins_spec missing spectral specs for lasers: {missing}")

    return by_laser


# =============================================================================
# Misc helpers
# =============================================================================

def _make_jkey_series(df: pd.DataFrame, fields: Sequence[str], *, sep: str) -> pd.Series:
    parts = []
    for f in fields:
        if f in df.columns:
            parts.append(df[f].astype(str).fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    jkey = parts[0]
    for p in parts[1:]:
        jkey = jkey + sep + p
    return jkey.str.replace(r"\s+", "", regex=True)


def _write_by_jkey_summary(df: pd.DataFrame, by_jkey_dir: Path, *, encoding: str, verbose: bool) -> None:
    if "jkey" not in df.columns:
        raise FourierStateError("Cannot split by jkey: fourier_summary.csv missing column 'jkey'.")

    for jkey, g in df.groupby("jkey"):
        safe = _sanitize_filename(str(jkey)) or "UNK"
        out_path = by_jkey_dir / f"Fourier_Resumen_{safe}.csv"
        g.to_csv(out_path, index=False, encoding=encoding)
        if verbose:
            print(f"[OK] wrote: {out_path}  (rows={len(g)})")


def _safe_sort(df: pd.DataFrame, sort_by: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in sort_by if c in df.columns]
    if not cols:
        return df.reset_index(drop=True)
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def _require_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FourierStateError(f"{label} missing required columns: {missing}")


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FourierStateError(f"{label} not found: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FourierStateError(f"{label} must be an existing directory: {path}")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(path.read_text(encoding="latin-1"))


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _short_hash(s: str) -> str:
    return hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).hexdigest()


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _sanitize_filename(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        s = s.replace(ch, "")
    return s.strip()


def _policy_to_jsonable(policy: FourierPolicy) -> dict:
    return {
        "split_by_jkey": bool(policy.split_by_jkey),
        "jkey_fields": list(policy.jkey_fields),
        "jkey_sep": str(policy.jkey_sep),
        "redact_labels_if_blind": bool(policy.redact_labels_if_blind),
        "redacted_value": str(policy.redacted_value),
        "drop_archivo": bool(policy.drop_archivo),
        "drop_parquet_path": bool(policy.drop_parquet_path),
        "psd_first_required": bool(policy.psd_first_required),
        "allow_raw_fallback": bool(policy.allow_raw_fallback),
        "psd_cache_dir_override": str(policy.psd_cache_dir_override) if policy.psd_cache_dir_override else None,
        "energia_run_meta_override": str(policy.energia_run_meta_override) if policy.energia_run_meta_override else None,
        "fmin": float(policy.fmin),
        "fmax": float(policy.fmax) if policy.fmax is not None else None,
        "enforce_freq_grid_consistency": bool(policy.enforce_freq_grid_consistency),
        "freq_grid_rtol": float(policy.freq_grid_rtol),
        "freq_grid_atol": float(policy.freq_grid_atol),
        "exclude_dc": True,
        "alpha": float(policy.alpha),
        "clamp_negative_psd_to_zero": bool(policy.clamp_negative_psd_to_zero),
        "write_pmf_long": bool(policy.write_pmf_long),
        "pmf_long_max_modes": int(policy.pmf_long_max_modes) if policy.pmf_long_max_modes is not None else None,
        "csv_encoding": str(policy.csv_encoding),
        "sort_by": list(policy.sort_by),
        "std_ddof": int(policy.std_ddof),
    }


# =============================================================================
# CLI
# =============================================================================
def _parse_args(argv: Optional[Sequence[str]] = None):
    import argparse
    p = argparse.ArgumentParser(description="State FOURIER (fine modal spectral entropy), PSD-cache-first.")
    p.add_argument("--informational-queue-csv", required=True, help="Path to informational_queue.csv")
    p.add_argument("--bins-spec-json", required=True, help="Path to bins_spec.json")
    p.add_argument("--out-dir", required=True, help="Output directory (Reports/Level4_Informational/Fourier)")
    p.add_argument("--lasers", required=True, help="Comma-separated laser names, e.g. Laser_1,Laser_2,...")
    p.add_argument("--mode", default="", help='Execution mode: "blind" or "declared"')
    p.add_argument("--write-pmf-long", action="store_true", help="Write fourier_pmf_long.csv (can be huge)")
    p.add_argument("--pmf-long-max-modes", type=int, default=0, help="If >0, limit pmf_long rows per (mid,laser) to top energy modes")
    p.add_argument("--fmin", type=float, default=0.0, help="Minimum frequency included (same units as PSD freqs)")
    p.add_argument("--fmax", type=float, default=-1.0, help="Maximum frequency included; if <0 uses bins_spec default per laser")
    p.add_argument("--alpha", type=float, default=0.0, help="Pseudocount alpha added to each modal energy before normalization")
    p.add_argument("--allow-raw-fallback", action="store_true", help="Allow raw FFT fallback when PSD cache missing (NOT recommended)")
    p.add_argument("--psd-cache-dir", default="", help="Override PSD cache directory")
    p.add_argument("--energia-run-meta", default="", help="Override energia_run_meta.json path")
    p.add_argument("--no-split-by-jkey", action="store_true", help="Disable by_jkey outputs")
    p.add_argument("--no-grid-check", action="store_true", help="Disable frequency grid consistency check")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    lasers = [s.strip() for s in str(args.lasers).split(",") if s.strip()]
    if not lasers:
        print("[FAIL] --lasers is required and must contain at least one laser name.")
        return 2

    policy = FourierPolicy(
        split_by_jkey=not bool(args.no_split_by_jkey),
        write_pmf_long=bool(args.write_pmf_long),
        pmf_long_max_modes=(int(args.pmf_long_max_modes) if int(args.pmf_long_max_modes) > 0 else None),
        fmin=float(args.fmin),
        fmax=(None if float(args.fmax) < 0 else float(args.fmax)),
        alpha=float(args.alpha),
        allow_raw_fallback=bool(args.allow_raw_fallback),
        psd_cache_dir_override=(Path(args.psd_cache_dir) if str(args.psd_cache_dir).strip() else None),
        energia_run_meta_override=(Path(args.energia_run_meta) if str(args.energia_run_meta).strip() else None),
        psd_first_required=not bool(args.allow_raw_fallback),
        enforce_freq_grid_consistency=not bool(args.no_grid_check),
    )

    try:
        run_state_fourier(
            informational_queue_csv=Path(args.informational_queue_csv),
            bins_spec_json=Path(args.bins_spec_json),
            out_dir=Path(args.out_dir),
            lasers=lasers,
            mode=str(args.mode).strip().lower(),
            policy=policy,
            verbose=bool(args.verbose),
        )
        return 0
    except Exception as e:
        print("\n" + "!" * 72)
        print("[FAIL] state_fourier.py")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
