# state_energia.py
from __future__ import annotations

"""
State: ENERGIA (Spectral Energy)

This module computes the ENERGIA state for each measurement and each laser channel:
- Compute an averaged PSD (via rFFT) over a small number of blocks per file.
- Integrate the PSD into fixed spectral bands (freq_edges from bins_spec.json).
- Convert band energies into a PMF over bands.
- Compute Shannon entropy (bits) per laser channel: Hspec_<laser>.
- Aggregate across lasers: mu_energ, sigma_energ, norm2_energ.

It is designed to be:
- Contract-driven (consumes informational_queue.csv and bins_spec.json).
- Low-resource friendly (streaming / incremental writers, small arrays).
- Blind-safe (optional redaction of labels when mode == "blind").
- Fourier-ready (optional PSD cache written per (mid, laser) to reuse later).

Expected inputs
---------------
- informational_queue.csv:
    Must include at least: parquet_path.
    Recommended: mid, fecha, lab, archivo, color.
- bins_spec.json:
    Must include spectral bins per laser:
        spectral.freq_edges (strictly increasing)
        spectral.n_fft, spectral.blocks_per_file, spectral.window, spectral.fs (or defaults)
        spectral.freq_unit (informational only)
- Parquet files at parquet_path containing per-laser columns.

Outputs
-------
- energia_summary.csv (always)
- energia_pmf_long.csv (optional)
- energia_bands_long.csv (optional)
- energia_run_meta.json (always)
- by_jkey/ splits (optional)
- psd_cache/ (optional) for Fourier reuse

All code and console messages are in English.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import csv
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd


# =============================================================================
# Exceptions
# =============================================================================

class EnergiaStateError(RuntimeError):
    """Raised when ENERGIA state computation cannot be produced."""


# =============================================================================
# Policy / Artifacts
# =============================================================================

@dataclass(frozen=True)
class EnergiaPolicy:
    # Organization
    split_by_jkey: bool = True
    jkey_fields: Tuple[str, ...] = ("fecha", "lab")
    jkey_sep: str = "_"

    # Blind-mode safety
    redact_labels_if_blind: bool = True
    redacted_value: str = "REDACTED"
    # If True, do NOT write `archivo` column even in declared mode (safer).
    drop_archivo: bool = True

    # Spectral computation
    block_selection: str = "random"  # "random" (deterministic by mid+laser) or "uniform"
    detrend_mean: bool = True
    max_bad_fraction_per_block: float = 0.05  # replace non-finite with 0, but fail block if too many
    min_blocks_required: int = 1  # minimum blocks to accept a channel as valid
    pseudocount: float = 0.0  # Laplace-style smoothing on band energies

    # Outputs
    write_pmf_long: bool = False
    write_bands_long: bool = False
    long_format: str = "csv"  # "csv" only (parquet append is not portable on all machines)

    # Fourier-ready cache (optional)
    write_psd_cache: bool = False
    psd_cache_format: str = "npz"  # "npz" only

    # Stats
    std_ddof: int = 0  # 0 = population std
    csv_encoding: str = "utf-8"
    sort_by: Tuple[str, ...] = ("fecha", "lab", "mid")


@dataclass(frozen=True)
class EnergiaArtifacts:
    out_dir: Path
    energia_summary_csv: Path
    energia_pmf_long_path: Optional[Path]
    energia_bands_long_path: Optional[Path]
    by_jkey_dir: Optional[Path]
    psd_cache_dir: Optional[Path]
    run_meta_json: Path


# =============================================================================
# Public API
# =============================================================================



def run_state_energia(
    *,
    informational_queue_csv: Union[str, Path],
    bins_spec_json: Union[str, Path],
    out_dir: Union[str, Path],
    lasers: Sequence[str],
    mode: str = "",
    policy: EnergiaPolicy = EnergiaPolicy(),
    verbose: bool = True,
) -> EnergiaArtifacts:
    """
    Core runner: computes ENERGIA state from queue + bins_spec.

    Parameters
    ----------
    informational_queue_csv:
        Path to informational_queue.csv produced by the quality gate.
    bins_spec_json:
        Path to bins_spec.json produced by the informational bins block.
    out_dir:
        Output directory (usually Reports/Level4_Informational/Energia).
    lasers:
        List of laser channel names to process (e.g., ["L1","L2","L3","L4","L5","L6"]).
    mode:
        "blind" or "declared" (controls redaction if policy.redact_labels_if_blind).
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

    psd_cache_dir = out_dir / "psd_cache" if policy.write_psd_cache else None
    if psd_cache_dir is not None:
        psd_cache_dir.mkdir(parents=True, exist_ok=True)

    # Long outputs (streaming CSV)
    energia_pmf_long_path: Optional[Path] = None
    energia_bands_long_path: Optional[Path] = None
    pmf_writer = None
    bands_writer = None
    pmf_fh = None
    bands_fh = None

    if policy.write_pmf_long:
        energia_pmf_long_path = out_dir / "energia_pmf_long.csv"
        pmf_fh = open(energia_pmf_long_path, "w", newline="", encoding=policy.csv_encoding)
        pmf_writer = csv.DictWriter(
            pmf_fh,
            fieldnames=["mid", "jkey", "laser", "band_index", "p", "E_band"],
        )
        pmf_writer.writeheader()

    if policy.write_bands_long:
        energia_bands_long_path = out_dir / "energia_bands_long.csv"
        bands_fh = open(energia_bands_long_path, "w", newline="", encoding=policy.csv_encoding)
        bands_writer = csv.DictWriter(
            bands_fh,
            fieldnames=["mid", "jkey", "laser", "band_index", "f_lo", "f_hi", "E_band"],
        )
        bands_writer.writeheader()

    bins_spec = _read_json(bpath)
    bins_hash = _sha256_bytes(bpath.read_bytes())
    spec_map = _extract_spectral_specs(bins_spec, lasers)

    # Queue
    queue = _read_queue(qpath)
    queue = _normalize_queue_columns(queue)
    _require_columns(queue, ["parquet_path"], "informational_queue.csv")
    queue["_jkey"] = _make_jkey_series(queue, policy.jkey_fields, sep=policy.jkey_sep)

    # Determine mode
    mode = (mode or "").strip().lower()
    if not mode and "mode" in bins_spec:
        mode = str(bins_spec.get("mode", "")).strip().lower()

    if verbose:
        print("\n" + "=" * 72)
        print("[Level4] Building ENERGIA state (Spectral energy entropy)")
        print("=" * 72)
        print(f"[IN ] queue     : {qpath}")
        print(f"[IN ] bins_spec : {bpath}  (sha256={bins_hash[:12]}...)")
        print(f"[OUT] dir       : {out_dir}")
        print(f"[MODE] {mode or 'unknown'}")
        print(f"[LASERS] {list(lasers)}")
        if policy.write_pmf_long:
            print(f"[OUT] energia_pmf_long.csv : {energia_pmf_long_path}")
        if policy.write_bands_long:
            print(f"[OUT] energia_bands_long.csv : {energia_bands_long_path}")
        if psd_cache_dir is not None:
            print(f"[OUT] psd_cache/ : {psd_cache_dir}")

    # Process measurement-by-measurement (low memory)
    summary_rows: List[Dict[str, Any]] = []
    n_fail = 0
    n_partial = 0
    n_ok = 0

    for idx, row in queue.iterrows():
        mid = _safe_str(row.get("mid")) or f"row{idx}"
        jkey = _safe_str(row.get("_jkey"))
        parquet_path = Path(_safe_str(row.get("parquet_path")))

        meta = _extract_meta(row, keep_archivo=not policy.drop_archivo)
        if policy.redact_labels_if_blind and mode == "blind":
            meta = _redact_meta(meta, redacted_value=policy.redacted_value)

        if not parquet_path.exists():
            summary_rows.append(_fail_row(mid, jkey, meta, reason=f"parquet not found: {parquet_path}"))
            n_fail += 1
            continue

        try:
            df = pd.read_parquet(parquet_path, columns=list(lasers))
        except Exception as e:
            summary_rows.append(_fail_row(mid, jkey, meta, reason=f"failed to read parquet: {type(e).__name__}: {e}"))
            n_fail += 1
            continue

        # Compute per laser
        Hspec: Dict[str, float] = {}
        Etot: Dict[str, float] = {}
        Nblocks: Dict[str, int] = {}
        Nfft: Dict[str, int] = {}

        valid_H: List[float] = []
        n_valid_channels = 0

        for laser in lasers:
            if laser not in df.columns:
                Hspec[laser] = np.nan
                Etot[laser] = np.nan
                Nblocks[laser] = 0
                Nfft[laser] = int(spec_map[laser]["n_fft"])
                continue

            x = df[laser].to_numpy(dtype=float, copy=False)
            spec = spec_map[laser]
            n_fft = int(spec["n_fft"])
            fs = float(spec["fs"])
            blocks_per_file = int(spec["blocks_per_file"])
            window_name = str(spec["window"])

            Nfft[laser] = n_fft

            out = _compute_energy_state_for_channel(
                x=x,
                mid=mid,
                laser=laser,
                freq_edges=np.asarray(spec["freq_edges"], dtype=float),
                n_fft=n_fft,
                fs=fs,
                blocks_per_file=blocks_per_file,
                window_name=window_name,
                selection=policy.block_selection,
                detrend_mean=policy.detrend_mean,
                max_bad_fraction_per_block=float(policy.max_bad_fraction_per_block),
                min_blocks_required=int(policy.min_blocks_required),
                pseudocount=float(policy.pseudocount),
                psd_cache_dir=psd_cache_dir,
                write_psd_cache=bool(policy.write_psd_cache),
                psd_cache_format=str(policy.psd_cache_format),
            )

            Hspec[laser] = out["Hspec"]
            Etot[laser] = out["E_tot"]
            Nblocks[laser] = out["n_blocks_used"]

            if out["ok"]:
                n_valid_channels += 1
                valid_H.append(out["Hspec"])

            # Stream long rows (optional)
            if pmf_writer is not None and out.get("pmf_rows") is not None:
                for r in out["pmf_rows"]:
                    pmf_writer.writerow(
                        {
                            "mid": mid,
                            "jkey": jkey,
                            "laser": laser,
                            "band_index": int(r["band_index"]),
                            "p": float(r["p"]),
                            "E_band": float(r["E_band"]),
                        }
                    )

            if bands_writer is not None and out.get("band_rows") is not None:
                for r in out["band_rows"]:
                    bands_writer.writerow(
                        {
                            "mid": mid,
                            "jkey": jkey,
                            "laser": laser,
                            "band_index": int(r["band_index"]),
                            "f_lo": float(r["f_lo"]),
                            "f_hi": float(r["f_hi"]),
                            "E_band": float(r["E_band"]),
                        }
                    )

        # Aggregate
        H_arr = np.array(valid_H, dtype=float)
        mu = float(np.nanmean(H_arr)) if H_arr.size > 0 else np.nan
        sigma = float(np.nanstd(H_arr, ddof=int(policy.std_ddof))) if H_arr.size > 1 else np.nan
        norm2 = float(np.sqrt(np.nansum(H_arr * H_arr))) if H_arr.size > 0 else np.nan

        status = "OK" if n_valid_channels == len(lasers) else ("PARTIAL" if n_valid_channels > 0 else "FAIL")
        if status == "OK":
            n_ok += 1
        elif status == "PARTIAL":
            n_partial += 1
        else:
            n_fail += 1

        out_row: Dict[str, Any] = {}
        out_row.update(meta)
        out_row["mid"] = mid
        out_row["jkey"] = jkey
        out_row["status"] = status
        out_row["n_valid_channels"] = int(n_valid_channels)
        out_row["mu_energ"] = mu
        out_row["sigma_energ"] = sigma
        out_row["norm2_energ"] = norm2

        for laser in lasers:
            out_row[f"Hspec_{laser}"] = float(Hspec.get(laser, np.nan)) if np.isfinite(Hspec.get(laser, np.nan)) else np.nan
            out_row[f"E_tot_{laser}"] = float(Etot.get(laser, np.nan)) if np.isfinite(Etot.get(laser, np.nan)) else np.nan
            out_row[f"nfft_{laser}"] = int(Nfft.get(laser, 0))
            out_row[f"nblocks_{laser}"] = int(Nblocks.get(laser, 0))

        summary_rows.append(out_row)

    # Close streaming handles
    if pmf_fh is not None:
        pmf_fh.close()
    if bands_fh is not None:
        bands_fh.close()

    # Summary DF
    summary = pd.DataFrame(summary_rows)
    summary = _safe_sort(summary, policy.sort_by)
    energia_summary_csv = out_dir / "energia_summary.csv"
    summary.to_csv(energia_summary_csv, index=False, encoding=policy.csv_encoding)

    # by_jkey splits
    if by_jkey_dir is not None:
        _write_by_jkey_summary(summary, by_jkey_dir, encoding=policy.csv_encoding, verbose=verbose)

    # Meta
    run_meta = {
        "module": "state_energia.py",
        "mode": mode or "",
        "bins_spec_sha256": bins_hash,
        "bins_spec_path": str(bpath),
        "queue_path": str(qpath),
        "outputs": {
            "out_dir": str(out_dir),
            "energia_summary_csv": str(energia_summary_csv),
            "energia_pmf_long_csv": str(energia_pmf_long_path) if energia_pmf_long_path else None,
            "energia_bands_long_csv": str(energia_bands_long_path) if energia_bands_long_path else None,
            "by_jkey_dir": str(by_jkey_dir) if by_jkey_dir else None,
            "psd_cache_dir": str(psd_cache_dir) if psd_cache_dir else None,
        },
        "counts": {
            "n_measurements": int(len(summary)),
            "n_ok": int(n_ok),
            "n_partial": int(n_partial),
            "n_fail": int(n_fail),
            "n_lasers": int(len(lasers)),
        },
        "policy": _policy_to_jsonable(policy),
        "spectral_specs": {
            laser: {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in spec_map[laser].items() if k != "freq_edges"}
            for laser in lasers
        },
    }
    run_meta_json = out_dir / "energia_run_meta.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding=policy.csv_encoding)

    if verbose:
        print(f"[OK] wrote: {energia_summary_csv}  (rows={len(summary)})")
        if energia_pmf_long_path:
            print(f"[OK] wrote: {energia_pmf_long_path}")
        if energia_bands_long_path:
            print(f"[OK] wrote: {energia_bands_long_path}")
        print(f"[OK] wrote: {run_meta_json}")

    return EnergiaArtifacts(
        out_dir=out_dir,
        energia_summary_csv=energia_summary_csv,
        energia_pmf_long_path=energia_pmf_long_path,
        energia_bands_long_path=energia_bands_long_path,
        by_jkey_dir=by_jkey_dir,
        psd_cache_dir=psd_cache_dir,
        run_meta_json=run_meta_json,
    )


# =============================================================================
# Core computations
# =============================================================================

def _compute_energy_state_for_channel(
    *,
    x: np.ndarray,
    mid: str,
    laser: str,
    freq_edges: np.ndarray,
    n_fft: int,
    fs: float,
    blocks_per_file: int,
    window_name: str,
    selection: str,
    detrend_mean: bool,
    max_bad_fraction_per_block: float,
    min_blocks_required: int,
    pseudocount: float,
    psd_cache_dir: Optional[Path],
    write_psd_cache: bool,
    psd_cache_format: str,
) -> Dict[str, Any]:
    """
    Compute PSD average, band energies, PMF, entropy for a single channel.
    Returns dict with:
      - ok (bool)
      - Hspec (float)
      - E_tot (float)
      - n_blocks_used (int)
      - pmf_rows (optional list)
      - band_rows (optional list)
    """
    out: Dict[str, Any] = {
        "ok": False,
        "Hspec": np.nan,
        "E_tot": np.nan,
        "n_blocks_used": 0,
        "pmf_rows": None,
        "band_rows": None,
    }

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)

    # Ensure edges valid
    freq_edges = np.asarray(freq_edges, dtype=float)
    if freq_edges.size < 2 or not np.all(np.isfinite(freq_edges)):
        return out
    if not np.all(np.diff(freq_edges) > 0):
        return out

    if n_fft <= 0:
        return out
    if fs <= 0:
        fs = 1.0

    if x.size < n_fft:
        return out

    # Prepare window
    w = _get_window(window_name, n_fft)
    if w is None:
        w = np.ones(n_fft, dtype=float)

    starts = _select_block_starts(
        n=x.size,
        n_fft=n_fft,
        blocks=blocks_per_file,
        seed=_stable_seed(mid, laser),
        method=selection,
    )
    if not starts:
        return out

    # Accumulate PSD average
    psd_sum = None
    n_used = 0
    bad_blocks = 0

    for s in starts:
        block = x[s : s + n_fft].astype(float, copy=False)

        bad = ~np.isfinite(block)
        if bad.any():
            frac = float(bad.mean())
            if frac > max_bad_fraction_per_block:
                bad_blocks += 1
                continue
            block = block.copy()
            block[bad] = 0.0

        if detrend_mean:
            block = block - float(np.mean(block))

        block = block * w

        # rFFT power
        Xf = np.fft.rfft(block, n=n_fft)
        P = (np.abs(Xf) ** 2) / float(n_fft)

        if psd_sum is None:
            psd_sum = P
        else:
            psd_sum += P
        n_used += 1

    if psd_sum is None or n_used < int(min_blocks_required):
        return out

    psd_avg = psd_sum / float(n_used)

    # Cache PSD for Fourier reuse (optional)
    if write_psd_cache and psd_cache_dir is not None and psd_cache_format.lower() == "npz":
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
        _write_psd_cache_npz(
            psd_cache_dir=psd_cache_dir,
            mid=mid,
            laser=laser,
            freqs=freqs.astype(np.float32),
            psd=psd_avg.astype(np.float32),
            meta={
                "n_fft": int(n_fft),
                "fs": float(fs),
                "window": str(window_name),
                "n_blocks_used": int(n_used),
                "selection": str(selection),
                "starts": starts,
            },
        )

    # Band energies
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    E_band, band_bounds = _integrate_psd_into_bands(psd_avg, freqs, freq_edges)

    E_tot = float(np.sum(E_band))
    out["E_tot"] = E_tot
    out["n_blocks_used"] = int(n_used)

    if not np.isfinite(E_tot) or E_tot <= 0.0:
        return out

    # PMF over bands
    K = int(E_band.size)
    alpha = float(pseudocount)
    if alpha > 0.0:
        p = (E_band + alpha) / (E_tot + alpha * K)
    else:
        p = E_band / E_tot

    # Shannon entropy in bits
    H = _shannon_entropy_bits(p)
    out["Hspec"] = float(H) if np.isfinite(H) else np.nan
    out["ok"] = bool(np.isfinite(out["Hspec"]))

    # Long rows (optional) are created by caller (streaming) to avoid memory use.
    out["pmf_rows"] = [
        {"band_index": k, "p": float(p[k]), "E_band": float(E_band[k])}
        for k in range(K)
    ]
    out["band_rows"] = [
        {"band_index": k, "f_lo": float(band_bounds[k][0]), "f_hi": float(band_bounds[k][1]), "E_band": float(E_band[k])}
        for k in range(K)
    ]

    return out


def _integrate_psd_into_bands(psd: np.ndarray, freqs: np.ndarray, freq_edges: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Integrate a PSD (non-negative) into band energies defined by freq_edges.

    Band convention:
      - all bands are [lo, hi) except the last which is [lo, hi] (inclusive).
    """
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    edges = np.asarray(freq_edges, dtype=float)

    K = edges.size - 1
    E = np.zeros(K, dtype=float)
    bounds: List[Tuple[float, float]] = []

    for k in range(K):
        lo = float(edges[k])
        hi = float(edges[k + 1])
        bounds.append((lo, hi))

        lo_idx = int(np.searchsorted(freqs, lo, side="left"))
        if k == K - 1:
            hi_idx = int(np.searchsorted(freqs, hi, side="right"))
        else:
            hi_idx = int(np.searchsorted(freqs, hi, side="left"))

        if hi_idx <= lo_idx:
            E[k] = 0.0
        else:
            E[k] = float(np.sum(psd[lo_idx:hi_idx]))

    return E, bounds


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

    # Canonicalize common aliases
    rename = {}
    if "laboratorio" in out.columns and "lab" not in out.columns:
        rename["laboratorio"] = "lab"
    if "measurement_id" in out.columns and "mid" not in out.columns:
        rename["measurement_id"] = "mid"
    if "path" in out.columns and "parquet_path" not in out.columns:
        rename["path"] = "parquet_path"
    if rename:
        out = out.rename(columns=rename)

    # Ensure mid exists without reviving legacy structural fields
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

def _extract_meta(row: pd.Series, *, keep_archivo: bool) -> Dict[str, Any]:
    keep = ["fecha", "lab", "color"]
    if keep_archivo:
        keep.append("archivo")
    meta = {}
    for k in keep:
        if k in row.index:
            meta[k] = _safe_str(row.get(k))
    return meta


def _redact_meta(meta: Dict[str, Any], *, redacted_value: str) -> Dict[str, Any]:
    out = dict(meta)
    for k in ("color", "archivo"):
        if k in out and out[k] not in ("", None):
            out[k] = redacted_value
    return out


def _fail_row(mid: str, jkey: str, meta: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
    out = dict(meta)
    out["mid"] = mid
    out["jkey"] = jkey
    out["status"] = "FAIL"
    out["n_valid_channels"] = 0
    out["mu_energ"] = np.nan
    out["sigma_energ"] = np.nan
    out["norm2_energ"] = np.nan
    out["reason"] = reason
    return out


def _extract_spectral_specs(bins_spec: Dict[str, Any], lasers: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-laser spectral spec from bins_spec.json.

    Accepted shapes:
      - {"bins": [{"laser": "L1", "spectral": {...}}, ...]}
      - {"channels": [{"laser": "L1", "spectral": {...}}, ...]}
    """
    channels = None
    if isinstance(bins_spec.get("bins"), list):
        channels = bins_spec.get("bins")
    elif isinstance(bins_spec.get("channels"), list):
        channels = bins_spec.get("channels")

    if not channels:
        raise EnergiaStateError("bins_spec.json must contain a list at key 'bins' or 'channels'")

    by_laser: Dict[str, Dict[str, Any]] = {}
    for ch in channels:
        laser = _safe_str(ch.get("laser"))
        if not laser:
            continue
        if laser not in lasers:
            continue

        spectral = ch.get("spectral")
        if not isinstance(spectral, dict):
            raise EnergiaStateError(f"bins_spec missing spectral section for laser={laser}")

        freq_edges = spectral.get("freq_edges")
        if freq_edges is None:
            raise EnergiaStateError(f"bins_spec spectral.freq_edges missing for laser={laser}")

        freq_edges_arr = np.asarray(freq_edges, dtype=float)
        if freq_edges_arr.size < 2 or not np.all(np.isfinite(freq_edges_arr)):
            raise EnergiaStateError(f"bins_spec spectral.freq_edges invalid for laser={laser}")
        if not np.all(np.diff(freq_edges_arr) > 0):
            raise EnergiaStateError(f"bins_spec spectral.freq_edges not strictly increasing for laser={laser}")

        n_fft = int(spectral.get("n_fft", 4096))
        blocks_per_file = int(spectral.get("blocks_per_file", 8))
        window = str(spectral.get("window", "hann"))
        fs = float(spectral.get("fs", 1.0))

        by_laser[laser] = {
            "freq_edges": freq_edges_arr,
            "n_fft": n_fft,
            "blocks_per_file": blocks_per_file,
            "window": window,
            "fs": fs,
            "freq_unit": str(spectral.get("freq_unit", "")),
            "method": str(spectral.get("method", "rfft_psd")),
        }

    missing = [l for l in lasers if l not in by_laser]
    if missing:
        raise EnergiaStateError(f"bins_spec missing spectral specs for lasers: {missing}")

    return by_laser


# =============================================================================
# Block selection and window
# =============================================================================

def _select_block_starts(*, n: int, n_fft: int, blocks: int, seed: int, method: str) -> List[int]:
    max_start = n - n_fft
    if max_start < 0:
        return []
    blocks = int(max(1, blocks))
    method = (method or "").strip().lower()

    if max_start == 0:
        return [0]

    if method == "uniform":
        if blocks == 1:
            return [max_start // 2]
        starts = np.linspace(0, max_start, blocks).round().astype(int).tolist()
        return [int(s) for s in starts]

    # Default: deterministic random
    rng = np.random.default_rng(seed)
    space = max_start + 1
    if blocks >= space:
        return list(range(space))
    # blocks is small (e.g., 8), choice without replacement is safe and efficient
    starts = rng.choice(space, size=blocks, replace=False)
    starts = np.sort(starts).astype(int).tolist()
    return [int(s) for s in starts]


def _get_window(name: str, n_fft: int) -> Optional[np.ndarray]:
    name = (name or "").strip().lower()
    if name in ("hann", "hanning"):
        return np.hanning(n_fft).astype(float)
    if name == "hamming":
        return np.hamming(n_fft).astype(float)
    if name == "blackman":
        return np.blackman(n_fft).astype(float)
    if name in ("boxcar", "rect", "rectangular", ""):
        return np.ones(n_fft, dtype=float)
    # Unknown: fallback to ones
    return np.ones(n_fft, dtype=float)


# =============================================================================
# PSD cache (Fourier-ready)
# =============================================================================

def _write_psd_cache_npz(
    *,
    psd_cache_dir: Path,
    mid: str,
    laser: str,
    freqs: np.ndarray,
    psd: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    # filename-safe
    mid_s = _sanitize_filename(mid) or "mid"
    laser_s = _sanitize_filename(laser) or "laser"
    path = psd_cache_dir / f"psd_{mid_s}_{laser_s}.npz"
    np.savez_compressed(path, freqs=freqs, psd=psd, meta=json.dumps(meta))


def load_psd_cache_npz(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Helper for future modules (e.g., state_fourier): load an npz PSD cache.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    z = np.load(path, allow_pickle=False)
    meta = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {}
    return {
        "freqs": z["freqs"],
        "psd": z["psd"],
        "meta": meta,
    }


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
        raise EnergiaStateError("Cannot split by jkey: energia_summary.csv missing column 'jkey'.")

    for jkey, g in df.groupby("jkey"):
        safe = _sanitize_filename(str(jkey)) or "UNK"
        out_path = by_jkey_dir / f"Energia_Resumen_{safe}.csv"
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
        raise EnergiaStateError(f"{label} missing required columns: {missing}")


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise EnergiaStateError(f"{label} not found: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise EnergiaStateError(f"{label} must be an existing directory: {path}")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(path.read_text(encoding="latin-1"))


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _stable_seed(mid: str, laser: str) -> int:
    # stable across runs and machines
    h = hashlib.blake2b((mid + "|" + laser).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)


def _short_hash(s: str) -> str:
    h = hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).hexdigest()
    return h


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


def _policy_to_jsonable(policy: EnergiaPolicy) -> Dict[str, Any]:
    return {
        "split_by_jkey": bool(policy.split_by_jkey),
        "jkey_fields": list(policy.jkey_fields),
        "jkey_sep": str(policy.jkey_sep),
        "redact_labels_if_blind": bool(policy.redact_labels_if_blind),
        "redacted_value": str(policy.redacted_value),
        "drop_archivo": bool(policy.drop_archivo),
        "block_selection": str(policy.block_selection),
        "detrend_mean": bool(policy.detrend_mean),
        "max_bad_fraction_per_block": float(policy.max_bad_fraction_per_block),
        "min_blocks_required": int(policy.min_blocks_required),
        "pseudocount": float(policy.pseudocount),
        "write_pmf_long": bool(policy.write_pmf_long),
        "write_bands_long": bool(policy.write_bands_long),
        "long_format": str(policy.long_format),
        "write_psd_cache": bool(policy.write_psd_cache),
        "psd_cache_format": str(policy.psd_cache_format),
        "std_ddof": int(policy.std_ddof),
        "csv_encoding": str(policy.csv_encoding),
        "sort_by": list(policy.sort_by),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args(argv: Optional[Sequence[str]] = None):
    import argparse
    p = argparse.ArgumentParser(description="State ENERGIA (Spectral energy entropy).")
    p.add_argument("--queue", required=True, help="Path to informational_queue.csv")
    p.add_argument("--bins-spec", required=True, help="Path to bins_spec.json")
    p.add_argument("--out-dir", required=True, help="Output directory (Reports/Level4_Informational/Energia)")
    p.add_argument("--lasers", required=True, help="Comma-separated lasers, e.g. L1,L2,L3,L4,L5,L6")
    p.add_argument("--mode", default="", help="blind|declared (affects redaction)")
    p.add_argument("--no-split-by-jkey", action="store_true", help="Disable by_jkey outputs")
    p.add_argument("--write-pmf-long", action="store_true", help="Write energia_pmf_long.csv (streaming)")
    p.add_argument("--write-bands-long", action="store_true", help="Write energia_bands_long.csv (streaming)")
    p.add_argument("--write-psd-cache", action="store_true", help="Write PSD cache npz files for Fourier reuse")
    p.add_argument("--block-selection", default="random", choices=["random", "uniform"], help="Block selection strategy")
    p.add_argument("--pseudocount", type=float, default=0.0, help="Pseudocount for PMF smoothing")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    lasers = tuple([x.strip() for x in str(args.lasers).split(",") if x.strip()])
    if not lasers:
        print("[FAIL] --lasers is empty")
        return 2

    policy = EnergiaPolicy(
        split_by_jkey=not bool(args.no_split_by_jkey),
        write_pmf_long=bool(args.write_pmf_long),
        write_bands_long=bool(args.write_bands_long),
        write_psd_cache=bool(args.write_psd_cache),
        block_selection=str(args.block_selection),
        pseudocount=float(args.pseudocount),
    )

    try:
        run_state_energia(
            informational_queue_csv=args.queue,
            bins_spec_json=args.bins_spec,
            out_dir=args.out_dir,
            lasers=lasers,
            mode=str(args.mode),
            policy=policy,
            verbose=bool(args.verbose),
        )
        return 0
    except Exception as e:
        print("\n" + "!" * 72)
        print("[FAIL] state_energia.py")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
