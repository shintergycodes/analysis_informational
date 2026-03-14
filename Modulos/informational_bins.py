# informational_bins.py
from __future__ import annotations

"""
Level 4 — Informational bin specification builder (single module)

This module builds ONE frozen bins contract (bins_spec.json + bins_summary.csv) that includes:
  - amplitude bins (per laser)
  - increment bins (per laser)
  - coupling categories (per laser, for mutual information)
  - spectral bands (per laser, for energy/fourier states)

Key design goals:
  - Low-resource friendly: read only required parquet columns; bounded in-memory sampling.
  - Reproducible: deterministic RNG seed.
  - Auditable: explicit schema_version + reference selection recorded.
  - Data-driven partitions: number of bins (B_amp, B_dx, Q_coupling, K_spectral) computed from data.

No printing/logging: orchestration and console output belong to main_v2.py.
"""

from dataclasses import dataclass, asdict, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json
import math

import numpy as np
import pandas as pd


# =============================================================================
# Exceptions
# =============================================================================

class InformationalBinsError(RuntimeError):
    """Raised when the Level-4 bins contract cannot be built consistently."""


# =============================================================================
# Policies (tunable but safe defaults)
# =============================================================================

@dataclass(frozen=True)
class QuantileEdgesPolicy:
    """
    Policy for building edges via empirical quantiles and choosing the number of bins from data.

    Partition count selection
    -------------------------
    We use Freedman–Diaconis (FD) to derive the number of bins from the data scale:
        h = 2 * IQR * n^{-1/3}
        B = ceil((hi - lo) / h)
    where (lo, hi) are robust trimmed endpoints.

    Then we compute B+1 quantiles to build *equiprobable* bins (more stable than equal-width
    for heavy-tailed data), and finally merge too-narrow bins using a MAD-based floor.
    """
    min_bins: int = 32
    max_bins: int = 256

    # Robust support trimming for defining (lo, hi)
    trim_q: float = 5e-4

    # Minimum bin width floor: min_width = kappa * sigma_mad
    kappa_min_width: float = 0.25

    # When quantiles tie, enforce strictly increasing edges by adding epsilon.
    epsilon_tie: float = 1e-12

    # Sampling limits (low-resource): we sample per file and then cap globally.
    per_file_cap: int = 50_000
    global_cap: int = 300_000

    # Safety: minimum usable samples
    min_samples: int = 4096


@dataclass(frozen=True)
class IncrementsPolicy(QuantileEdgesPolicy):
    """
    Increment edges policy.

    Support for increments is defined symmetrically using a tail quantile of |dx|:
        L = quantile(|dx|, tail_q)
        support = [-L, +L]
    """
    tail_q: float = 0.999


@dataclass(frozen=True)
class CouplingPolicy:
    """
    Coupling categories policy (for mutual information).

    Partition count Q is chosen from data to keep joint tables well-populated:
        expected_count_per_cell = N / Q^2
    We choose the largest Q such that expected_count_per_cell >= min_expected_per_cell.
    """
    enabled: bool = True
    Q_min: int = 4
    Q_max: int = 16
    min_expected_per_cell: float = 25.0

    # Sampling caps (use amplitude samples)
    per_file_cap: int = 50_000
    global_cap: int = 300_000

    epsilon_tie: float = 1e-12


@dataclass(frozen=True)
class SpectralPolicy:
    """
    Spectral bands policy (for energy/fourier states).

    We compute a *control-average PSD* (Welch-style) per laser from short blocks and then
    pick frequency edges so that each band carries roughly equal average energy.

    Number of bands K is computed from data (frequency resolution):
        R = n_fft//2 + 1
        K = clamp(K_min, K_max, round(sqrt(R)))
    """
    enabled: bool = True

    # If fs_hz is None, we work in cycles_per_sample with fs = 1.0 (unitless).
    fs_hz: Optional[float] = None

    # PSD estimation
    n_fft: int = 4096
    blocks_per_file: int = 8
    max_files: int = 40
    window: str = "hann"  # hann, hamming, boxcar

    # Band count bounds
    K_min: int = 6
    K_max: int = 48

    epsilon_edge: float = 1e-12
    seed: int = 123456


@dataclass(frozen=True)
class BinsPolicy:
    """Top-level bins policy container."""
    amplitude: QuantileEdgesPolicy = field(default_factory=QuantileEdgesPolicy)
    increments: IncrementsPolicy = field(default_factory=IncrementsPolicy)
    coupling: CouplingPolicy = field(default_factory=CouplingPolicy)
    spectral: SpectralPolicy = field(default_factory=SpectralPolicy)

    seed: int = 123456


@dataclass(frozen=True)
class BinsArtifacts:
    bins_spec_json: Path
    bins_summary_csv: Path


# =============================================================================
# Public API
# =============================================================================

CfgLike = Union[Dict[str, Any], Any]


def build_bins_spec_from_level4_config_json(
    level4_config_json: Union[str, Path],
    *,
    policy: BinsPolicy = BinsPolicy(),
    parquet_engine: str = "auto",
) -> BinsArtifacts:
    """
    Convenience entrypoint: load level4_config.json and build bins artifacts.

    This supports running a standalone bins build while still using
    the exact same IO contract.
    """
    cfg = _read_json(Path(level4_config_json))
    return build_bins_spec_from_config(
        cfg,
        policy=policy,
        reference_dates=None,
        parquet_engine=parquet_engine,
    )

def build_bins_spec_from_config(
    level2_cfg: CfgLike,
    *,
    policy: BinsPolicy = BinsPolicy(),
    reference_dates: Optional[Sequence[str]] = None,
    parquet_engine: str = "auto",
) -> BinsArtifacts:
    """
    Build the unified bins contract from an InformationalConfig-like object OR a dict.

    Required fields:
      - queues.quality_scores_by_file  (Path-like)
      - level2_reports_dir             (Path-like)
      - lasers                         (Sequence[str])

    Optional fields:
      - bins_reference_group  ('ALL'|'UNK')
      - bins_reference_dates  (list[str])
      - mode                  ('blind'|'declared') (stored for auditing)
      - fs_hz / sampling_rate_hz (spectral interpretation)
    """
    target_root = _cfg_get(level2_cfg, ["target_root"], default=None)
    out_dir = Path(_cfg_get(level2_cfg, ["level2_reports_dir"], default=None))
    queues = _cfg_get(level2_cfg, ["queues"], default=None)
    quality_queue = _cfg_get(queues, ["quality_scores_by_file"], default=None)
    lasers = list(_cfg_get(level2_cfg, ["lasers"], default=[] ) or [])
    if not lasers:
        raise InformationalBinsError(
            "Level4 config has empty 'lasers'; cannot build per-laser bins."
        )
    if quality_queue is None:
        raise InformationalBinsError(
            "Level4 config missing queues.quality_scores_by_file."
        )

    if out_dir is None:
        raise InformationalBinsError("Level4 config missing level2_reports_dir.")
    quality_queue = _resolve_path(quality_queue, base_dir=target_root)

    ref_group = str(_cfg_get(level2_cfg, ["bins_reference_group"], default="ALL") or "ALL").strip()
    if reference_dates is None:
        reference_dates = _cfg_get(level2_cfg, ["bins_reference_dates"], default=None)

    mode = str(_cfg_get(level2_cfg, ["mode"], default="blind") or "blind").strip().lower()
    
    # If spectral fs_hz is not set in policy, try to infer it from config
    fs_candidate = _first_non_null(
        _cfg_get(level2_cfg, ["fs_hz"], default=None),
        _cfg_get(level2_cfg, ["sampling_rate_hz"], default=None),
        _cfg_get(level2_cfg, ["sampling_rate"], default=None),
    )
    if policy.spectral.fs_hz is None and fs_candidate is not None:
        try:
            fs_val = float(fs_candidate)
            if fs_val > 0:
                policy = replace(policy, spectral=replace(policy.spectral, fs_hz=fs_val))
        except Exception:
            pass  # keep None; unitless frequencies

    return build_bins_spec(
        informational_queue_csv=Path(quality_queue),
        out_dir=Path(out_dir),
        lasers=lasers,
        policy=policy,
        reference_group=ref_group,
        reference_dates=reference_dates,
        mode=mode,
        parquet_engine=parquet_engine,
    )


def build_bins_spec(
    *,
    informational_queue_csv: Path,
    out_dir: Path,
    lasers: Sequence[str],
    policy: BinsPolicy = BinsPolicy(),
    reference_group: str = "ALL",
    reference_dates: Optional[Sequence[str]] = None,
    mode: str = "blind",
    parquet_engine: str = "auto",
) -> BinsArtifacts:
    """
    Core builder.

    Outputs (written into out_dir):
      - bins_spec.json
      - bins_summary.csv
    """
    informational_queue_csv = Path(informational_queue_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not informational_queue_csv.exists():
        raise InformationalBinsError(f"informational_queue_csv not found: {informational_queue_csv}")

    qdf = _read_csv_usecols(
        informational_queue_csv,
        usecols=["fecha", "lab", "parquet_path"],
    )
    _require_cols(qdf, ["fecha", "lab", "parquet_path"], ctx=str(informational_queue_csv))

    ref = _select_reference_rows(
        qdf=qdf,
        reference_group=str(reference_group),
        reference_dates=reference_dates,
    )
    if ref.empty:
        raise InformationalBinsError(
            f"Reference set is empty for group={reference_group!r} and dates={reference_dates!r}."
        )

    ref_paths = [Path(str(p)) for p in ref["parquet_path"].tolist()]
    ref_paths = _dedupe_paths_preserve_order(ref_paths)
    if not ref_paths:
        raise InformationalBinsError("Reference parquet list is empty after deduplication.")

    for p in ref_paths:
        if not p.exists():
            raise InformationalBinsError(f"Reference parquet not found on disk: {p}")

    lasers = [str(x) for x in lasers]
    rng = np.random.default_rng(int(policy.seed))

    fs_hz, freq_unit = _resolve_sampling_rate(policy=policy)

    channels: List[Dict[str, Any]] = []
    legacy_bins: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    per_laser_amp_n_total: List[int] = []
    amp_samples_by_laser: Dict[str, np.ndarray] = {}

    # First pass: amplitude + increments
    for laser in lasers:
        amp_sample, amp_n_total = _collect_amplitude_sample(
            parquet_paths=ref_paths,
            laser_col=laser,
            per_file_cap=int(policy.amplitude.per_file_cap),
            global_cap=int(policy.amplitude.global_cap),
            rng=rng,
            parquet_engine=parquet_engine,
        )
        if amp_sample.size < int(policy.amplitude.min_samples) or amp_n_total < 2:
            raise InformationalBinsError(
                f"Insufficient amplitude samples for {laser}: sample={amp_sample.size}, total={amp_n_total}."
            )

        lo, hi = _trim_range(amp_sample, q=float(policy.amplitude.trim_q))
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise InformationalBinsError(f"Invalid amplitude support for {laser}: lo={lo}, hi={hi}")

        sigma_amp = _sigma_mad(amp_sample)
        if not np.isfinite(sigma_amp):
            raise InformationalBinsError(f"Invalid MAD sigma (amplitude) for {laser}: {sigma_amp}")

        B_amp = _choose_bins_fd(
            sample=amp_sample,
            n_total=amp_n_total,
            lo=lo,
            hi=hi,
            min_bins=int(policy.amplitude.min_bins),
            max_bins=int(policy.amplitude.max_bins),
        )

        amp_edges = _quantile_edges_from_sample(
            sample=amp_sample,
            n_bins=B_amp,
            lo=lo,
            hi=hi,
            epsilon=float(policy.amplitude.epsilon_tie),
        )

        min_w_amp = max(float(policy.amplitude.epsilon_tie), float(policy.amplitude.kappa_min_width) * float(sigma_amp))
        amp_edges = _merge_min_width_edges_preserve_endpoints(
            edges=amp_edges,
            lo=lo,
            hi=hi,
            min_width=min_w_amp,
            epsilon=float(policy.amplitude.epsilon_tie),
        )
        B_amp_final = int(len(amp_edges) - 1)
        if B_amp_final < 2:
            raise InformationalBinsError(f"Amplitude bins collapsed for {laser}; consider lowering min_bins or kappa.")

        # increments
        dx_sample, dx_n_total = _collect_increment_sample(
            parquet_paths=ref_paths,
            laser_col=laser,
            per_file_cap=int(policy.increments.per_file_cap),
            global_cap=int(policy.increments.global_cap),
            rng=rng,
            parquet_engine=parquet_engine,
        )
        if dx_sample.size < max(256, int(policy.increments.min_samples // 4)) or dx_n_total < 2:
            raise InformationalBinsError(
                f"Insufficient increment samples for {laser}: sample={dx_sample.size}, total={dx_n_total}."
            )

        sigma_dx = _sigma_mad(dx_sample)
        if not np.isfinite(sigma_dx):
            raise InformationalBinsError(f"Invalid MAD sigma (increments) for {laser}: {sigma_dx}")

        L = float(np.quantile(np.abs(dx_sample[np.isfinite(dx_sample)]), float(policy.increments.tail_q)))
        if not (np.isfinite(L) and L > 0):
            L = 6.0 * float(sigma_dx) if (np.isfinite(sigma_dx) and sigma_dx > 0) else float(np.max(np.abs(dx_sample)))
        if not (np.isfinite(L) and L > 0):
            raise InformationalBinsError(f"Invalid increment tail support for {laser}: L={L}")

        dx_lo, dx_hi = -L, +L

        B_dx = _choose_bins_fd(
            sample=dx_sample,
            n_total=dx_n_total,
            lo=dx_lo,
            hi=dx_hi,
            min_bins=int(policy.increments.min_bins),
            max_bins=int(policy.increments.max_bins),
        )

        dx_edges = _quantile_edges_from_sample(
            sample=dx_sample,
            n_bins=B_dx,
            lo=dx_lo,
            hi=dx_hi,
            epsilon=float(policy.increments.epsilon_tie),
        )

        min_w_dx = max(float(policy.increments.epsilon_tie), float(policy.increments.kappa_min_width) * float(sigma_dx))
        dx_edges = _merge_min_width_edges_preserve_endpoints(
            edges=dx_edges,
            lo=dx_lo,
            hi=dx_hi,
            min_width=min_w_dx,
            epsilon=float(policy.increments.epsilon_tie),
        )
        B_dx_final = int(len(dx_edges) - 1)
        if B_dx_final < 2:
            raise InformationalBinsError(f"Increment bins collapsed for {laser}; consider lowering min_bins or kappa.")

        per_laser_amp_n_total.append(int(amp_n_total))
        amp_samples_by_laser[laser] = amp_sample

        ch = {
            "laser": laser,
            "amplitude_edges": [float(x) for x in amp_edges],
            "increment_edges": [float(x) for x in dx_edges],
            "stats": {
                "n_samples_used_amp": int(amp_sample.size),
                "n_samples_total_amp": int(amp_n_total),
                "n_samples_used_dx": int(dx_sample.size),
                "n_samples_total_dx": int(dx_n_total),
                "amp_support_lo": float(amp_edges[0]),
                "amp_support_hi": float(amp_edges[-1]),
                "dx_support_lo": float(dx_edges[0]),
                "dx_support_hi": float(dx_edges[-1]),
                "sigma_mad_amp": float(sigma_amp),
                "sigma_mad_dx": float(sigma_dx),
                "B_amp": int(B_amp_final),
                "B_dx": int(B_dx_final),
                "tail_q_dx": float(policy.increments.tail_q),
            },
        }
        channels.append(ch)

        legacy_bins.append(
            {
                "laser": laser,
                "amplitude": {
                    "B": int(B_amp_final),
                    "edges": [float(x) for x in amp_edges],
                    "clip_out_of_range": True,
                    "support_lo": float(amp_edges[0]),
                    "support_hi": float(amp_edges[-1]),
                    "sigma_mad": float(sigma_amp),
                    "min_width": float(min_w_amp),
                    "trim_q": float(policy.amplitude.trim_q),
                    "method": "quantile_edges_fd_B",
                },
                "increments": {
                    "B": int(B_dx_final),
                    "edges": [float(x) for x in dx_edges],
                    "clip_out_of_range": True,
                    "support_lo": float(dx_edges[0]),
                    "support_hi": float(dx_edges[-1]),
                    "sigma_mad": float(sigma_dx),
                    "min_width": float(min_w_dx),
                    "tail_q": float(policy.increments.tail_q),
                    "method": "quantile_edges_fd_B_symmetric_support",
                },
            }
        )

        summary_rows.append(
            {
                "laser": laser,
                "B_amp": int(B_amp_final),
                "B_dx": int(B_dx_final),
                "amp_lo": float(amp_edges[0]),
                "amp_hi": float(amp_edges[-1]),
                "dx_lo": float(dx_edges[0]),
                "dx_hi": float(dx_edges[-1]),
                "sigma_amp_mad": float(sigma_amp),
                "sigma_dx_mad": float(sigma_dx),
                "n_amp_total": int(amp_n_total),
                "n_dx_total": int(dx_n_total),
                "n_amp_used": int(amp_sample.size),
                "n_dx_used": int(dx_sample.size),
            }
        )
    #
    # Coupling edges (data-driven Q)
    if bool(policy.coupling.enabled):
        # IMPORTANT: use an "effective N" closer to what we actually sampled (low-resource bound)
        min_used_amp = int(min(int(c["stats"]["n_samples_used_amp"]) for c in channels)) if channels else 0

        Q_coupling = _choose_Q_coupling(
            n_total=min_used_amp,
            Q_min=int(policy.coupling.Q_min),
            Q_max=int(policy.coupling.Q_max),
            min_expected=float(policy.coupling.min_expected_per_cell),
        )

        Q_by_laser: Dict[str, int] = {}

        for ch in channels:
            laser = ch["laser"]
            amp_sample = amp_samples_by_laser[laser]
            lo = float(ch["stats"]["amp_support_lo"])
            hi = float(ch["stats"]["amp_support_hi"])

            coup_edges = _quantile_edges_from_sample(
                sample=amp_sample,
                n_bins=Q_coupling,
                lo=lo,
                hi=hi,
                epsilon=float(policy.coupling.epsilon_tie),
            )

            # NEW (CRITICAL): merge too-narrow coupling bins (handles quantized amplitudes / tied quantiles)
            sigma_amp = float(ch["stats"]["sigma_mad_amp"])
            # Reuse amplitude min-width logic for coupling thresholds
            min_w_coup = float(policy.amplitude.kappa_min_width) * sigma_amp if sigma_amp > 0 else 0.0
            #
            coup_edges = _merge_min_width_edges_preserve_endpoints(
                edges=coup_edges,
                lo=lo,
                hi=hi,
                min_width=max(min_w_coup, 1e-9),
                epsilon=float(policy.coupling.epsilon_tie),
            )

            ch["coupling_edges"] = [float(x) for x in coup_edges]
            q_eff = int(len(coup_edges) - 1)
            ch["stats"]["Q_coupling"] = q_eff
            Q_by_laser[laser] = q_eff

        for b in legacy_bins:
            laser = b["laser"]
            coup_edges = next(c for c in channels if c["laser"] == laser)["coupling_edges"]
            b["coupling_categories"] = {
                "Q": int(len(coup_edges) - 1),
                "edges": [float(x) for x in coup_edges],
                "clip_out_of_range": True,
                "support_lo": float(b["amplitude"]["support_lo"]),
                "support_hi": float(b["amplitude"]["support_hi"]),
                "method": "quantile_edges_data_driven_Q",
            }

        # NEW: write per-laser effective Q into bins_summary (avoid lying with the global Q_coupling)
        for row in summary_rows:
            laser = str(row.get("laser", ""))
            if laser in Q_by_laser:
                row["Q_coupling"] = int(Q_by_laser[laser])

###############################################
    # Spectral edges (equal-energy on control-average PSD)
    if bool(policy.spectral.enabled):
        fs_eff = float(fs_hz) if fs_hz is not None else 1.0
        rng_spec = np.random.default_rng(int(policy.spectral.seed))

        for ch in channels:
            laser = ch["laser"]
            edges, K = _build_spectral_edges_equal_energy(
                parquet_paths=ref_paths,
                laser_col=laser,
                fs=fs_eff,
                n_fft=int(policy.spectral.n_fft),
                blocks_per_file=int(policy.spectral.blocks_per_file),
                max_files=int(policy.spectral.max_files),
                window=str(policy.spectral.window),
                K_min=int(policy.spectral.K_min),
                K_max=int(policy.spectral.K_max),
                epsilon=float(policy.spectral.epsilon_edge),
                rng=rng_spec,
                parquet_engine=parquet_engine,
            )
            ch["spectral"] = {
                "freq_edges": [float(x) for x in edges],
                "K": int(K),
                "freq_unit": freq_unit,
                "fs": float(fs_eff),
                "n_fft": int(policy.spectral.n_fft),
                "blocks_per_file": int(policy.spectral.blocks_per_file),
                "window": str(policy.spectral.window),
                "method": "equal_energy_on_control_avg_psd",
            }
            ch["stats"]["K_spectral"] = int(K)

        for b in legacy_bins:
            laser = b["laser"]
            spec = next(c for c in channels if c["laser"] == laser)["spectral"]
            b["spectral"] = dict(spec)

        for row in summary_rows:
            row["freq_unit"] = freq_unit
            row["fs"] = float(fs_eff)
#############################################

    # Write spec + summary
    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    spec_v2 = {
        "schema_version": "2.0",
        "created_utc": created_utc,
        "kind": "informational_bins",
        "mode": str(mode),
        "reference": {
            "group": str(reference_group),
            "dates": [str(d) for d in reference_dates] if reference_dates is not None else None,
            "source_queue": str(informational_queue_csv),
            "n_reference_rows": int(ref.shape[0]),
            "n_reference_parquets": int(len(ref_paths)),
            "selection_rule": _reference_rule_text(reference_group=str(reference_group), reference_dates=reference_dates),
        },
        "policy": {
            "amplitude": asdict(policy.amplitude),
            "increments": asdict(policy.increments),
            "coupling": asdict(policy.coupling),
            "spectral": asdict(policy.spectral),
            "seed": int(policy.seed),
        },
        "lasers": list(lasers),
        "channels": channels,  # canonical v2
        "bins": legacy_bins,   # transitional compatibility
        "notes": [
            "Unified Level-4 bins contract: amplitude + increments + coupling + spectral.",
            "Partition counts are computed from data (FD for amplitude/increments; expected-cell constraint for coupling; sqrt(freq_resolution) for spectral).",
            "Legacy 'bins' key is included temporarily for compatibility; prefer 'channels' for new code.",
        ],
    }

    bins_spec_json = out_dir / "bins_spec.json"
    bins_summary_csv = out_dir / "bins_summary.csv"

    with bins_spec_json.open("w", encoding="utf-8") as f:
        json.dump(spec_v2, f, ensure_ascii=False, indent=2)

    pd.DataFrame(summary_rows).to_csv(bins_summary_csv, index=False, encoding="utf-8")

    return BinsArtifacts(bins_spec_json=bins_spec_json, bins_summary_csv=bins_summary_csv)


# =============================================================================
# Internals
# =============================================================================

def _first_non_null(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _cfg_get(cfg: Any, path: Sequence[str], default: Any = None) -> Any:
    """
    Safe nested getter supporting:
      - dicts: cfg[key]
      - objects: getattr(cfg, key)
    """
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _resolve_path(value: Any, base_dir: Optional[Any]) -> Path:
    """
    Resolve a possibly-relative path.
    If base_dir is provided and the path is not absolute, we resolve relative to base_dir.
    """
    p = Path(str(value))
    if p.is_absolute():
        return p
    if base_dir is None:
        return p
    return Path(str(base_dir)) / p


def _read_json(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        raise InformationalBinsError(f"JSON file not found: {path}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        raise InformationalBinsError(f"Failed to read JSON: {path} ({type(e).__name__}: {e})") from e


def _read_csv_usecols(path: Path, usecols: Sequence[str]) -> pd.DataFrame:
    """
    Robust CSV read for small control files (queues). Tries UTF-8 then latin-1.
    Only reads the requested columns (low RAM).
    """
    last_err = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, usecols=list(usecols), encoding=enc)
            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    raise InformationalBinsError(f"Failed to read CSV: {path} ({type(last_err).__name__}: {last_err})")


def _require_cols(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise InformationalBinsError(f"Missing columns {missing} in {ctx}. Available: {list(df.columns)}")


def _select_reference_rows(
    *,
    qdf: pd.DataFrame,
    reference_group: str,
    reference_dates: Optional[Sequence[str]],
) -> pd.DataFrame:
    """
    Select the reference set D from the quality queue under the active contract.

    Active rule:
      - if reference_group == 'ALL': all rows
      - otherwise: no structural filtering by legacy groups
      - optional: filter fecha in reference_dates
    """
    rg = str(reference_group).strip().upper()
    df = qdf.copy()
    df["fecha"] = df["fecha"].astype("string").str.strip()

    if rg not in ("ALL", "UNK", ""):
        raise InformationalBinsError(
            f"Unsupported reference_group under active contract: {reference_group!r}. "
            "Use 'ALL' or provide reference_dates explicitly."
        )

    if reference_dates is not None:
        dates = [str(d).strip() for d in reference_dates]
        df = df[df["fecha"].isin(dates)].copy()

    df = df.dropna(subset=["parquet_path"]).copy()
    return df


def _reference_rule_text(*, reference_group: str, reference_dates: Optional[Sequence[str]]) -> str:
    rg = str(reference_group).upper()
    if rg == "ALL":
        base = "use all rows"
    else:
        base = f"use active contract reference group {reference_group!r} without legacy jornada filtering"
    if reference_dates is None:
        return base
    return base + " AND fecha in provided dates"

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


def _read_parquet_column(path: Path, col: str, engine: str = "auto") -> np.ndarray:
    """
    Read a single column from a parquet file to reduce memory usage.
    Falls back to full read if the backend doesn't support 'columns'.
    """
    try:
        if engine == "auto":
            df = pd.read_parquet(path, columns=[col])
        else:
            df = pd.read_parquet(path, columns=[col], engine=engine)
    except TypeError:
        df = pd.read_parquet(path) if engine == "auto" else pd.read_parquet(path, engine=engine)
        if col not in df.columns:
            raise InformationalBinsError(f"Column {col!r} not found in parquet: {path}")
        df = df[[col]]
    except Exception as e:
        raise InformationalBinsError(f"Failed reading parquet: {path} ({type(e).__name__}: {e})") from e

    if col not in df.columns:
        raise InformationalBinsError(f"Column {col!r} not found in parquet: {path}")
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return x


def _collect_amplitude_sample(
    *,
    parquet_paths: Sequence[Path],
    laser_col: str,
    per_file_cap: int,
    global_cap: int,
    rng: np.random.Generator,
    parquet_engine: str,
) -> Tuple[np.ndarray, int]:
    """
    Bounded-memory sampling across reference parquets.

    Trade-off:
      - Not a perfect uniform sample across pooled values when files are imbalanced,
        but fast, bounded, and robust for stable quantile edges.
    """
    samples: List[np.ndarray] = []
    n_total = 0

    for p in parquet_paths:
        x = _read_parquet_column(p, laser_col, engine=parquet_engine)
        x = x[np.isfinite(x)]
        n_total += int(x.size)
        if x.size == 0:
            continue

        if x.size <= per_file_cap:
            take = x
        else:
            idx = rng.choice(x.size, size=int(per_file_cap), replace=False)
            take = x[idx]
        samples.append(take)

    if not samples:
        return np.asarray([], dtype=float), int(n_total)

    samp = np.concatenate(samples, axis=0)
    if samp.size > int(global_cap):
        idx = rng.choice(samp.size, size=int(global_cap), replace=False)
        samp = samp[idx]

    return samp.astype(float, copy=False), int(n_total)


def _collect_increment_sample(
    *,
    parquet_paths: Sequence[Path],
    laser_col: str,
    per_file_cap: int,
    global_cap: int,
    rng: np.random.Generator,
    parquet_engine: str,
) -> Tuple[np.ndarray, int]:
    """
    Build a finite increment sample from truly-adjacent finite pairs within each file:
      dx_t = x_{t+1} - x_t, only where both values are finite.
    """
    samples: List[np.ndarray] = []
    n_total = 0

    for p in parquet_paths:
        x = _read_parquet_column(p, laser_col, engine=parquet_engine)
        finite = np.isfinite(x)
        if x.size < 2 or not np.any(finite):
            continue

        ok = finite[:-1] & finite[1:]
        if not np.any(ok):
            continue

        dx = x[1:][ok] - x[:-1][ok]
        dx = dx[np.isfinite(dx)]
        n_total += int(dx.size)
        if dx.size == 0:
            continue

        if dx.size <= per_file_cap:
            take = dx
        else:
            idx = rng.choice(dx.size, size=int(per_file_cap), replace=False)
            take = dx[idx]
        samples.append(take)

    if not samples:
        return np.asarray([], dtype=float), int(n_total)

    samp = np.concatenate(samples, axis=0)
    if samp.size > int(global_cap):
        idx = rng.choice(samp.size, size=int(global_cap), replace=False)
        samp = samp[idx]

    return samp.astype(float, copy=False), int(n_total)


def _trim_range(sample: np.ndarray, q: float) -> Tuple[float, float]:
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    qq = float(q)
    if not (0.0 <= qq < 0.5):
        raise InformationalBinsError(f"trim_q must be in [0,0.5). Got {q}")
    lo = float(np.quantile(x, qq))
    hi = float(np.quantile(x, 1.0 - qq))
    return lo, hi


def _sigma_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _choose_bins_fd(
    *,
    sample: np.ndarray,
    n_total: int,
    lo: float,
    hi: float,
    min_bins: int,
    max_bins: int,
) -> int:
    """
    Choose number of bins using Freedman–Diaconis (data-driven).
    """
    n = int(max(1, n_total))
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        B = int(round(math.sqrt(n)))
        return int(max(min_bins, min(max_bins, max(2, B))))

    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = float(q75 - q25)
    span = float(hi - lo)

    if not (np.isfinite(iqr) and iqr > 0 and np.isfinite(span) and span > 0):
        B = int(round(math.sqrt(n)))
        return int(max(min_bins, min(max_bins, max(2, B))))

    h = 2.0 * iqr * (n ** (-1.0 / 3.0))
    if not (np.isfinite(h) and h > 0):
        B = int(round(math.sqrt(n)))
        return int(max(min_bins, min(max_bins, max(2, B))))

    B = int(math.ceil(span / h))
    B = max(2, B)
    return int(max(min_bins, min(max_bins, B)))


def _quantile_edges_from_sample(
    *,
    sample: np.ndarray,
    n_bins: int,
    lo: float,
    hi: float,
    epsilon: float,
) -> np.ndarray:
    """
    Compute quantile-based edges (equiprobable bins) on a clipped sample.
    Ensures strictly increasing edges by applying an epsilon tie-break.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise InformationalBinsError("Cannot compute quantiles from an empty sample.")
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise InformationalBinsError(f"Invalid support: lo={lo}, hi={hi}")

    B = int(max(2, int(n_bins)))
    x = np.clip(x, lo, hi)
    qs = np.linspace(0.0, 1.0, B + 1, dtype=float)
    edges = np.quantile(x, qs)

    edges[0] = float(lo)
    edges[-1] = float(hi)

    eps = float(epsilon)
    for k in range(1, edges.size):
        if edges[k] <= edges[k - 1]:
            edges[k] = edges[k - 1] + eps

    return edges.astype(float, copy=False)


def _merge_min_width_edges_preserve_endpoints(
    *,
    edges: np.ndarray,
    lo: float,
    hi: float,
    min_width: float,
    epsilon: float,
) -> np.ndarray:
    """
    Merge bins that would be narrower than min_width.
    Endpoints are preserved exactly as (lo, hi).
    """
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise InformationalBinsError("Edges must be a 1D array with at least 2 values.")
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise InformationalBinsError("Invalid (lo, hi) for edge merging.")
    mw = float(min_width)
    if not (np.isfinite(mw) and mw > 0):
        return e

    kept: List[float] = [float(lo)]
    for x in e[1:-1]:
        x = float(x)
        if x - kept[-1] >= mw:
            kept.append(x)
    kept.append(float(hi))

    out = np.asarray(kept, dtype=float)
    eps = float(epsilon)
    for k in range(1, out.size):
        if out[k] <= out[k - 1]:
            out[k] = out[k - 1] + eps
    return out


def _choose_Q_coupling(*, n_total: int, Q_min: int, Q_max: int, min_expected: float) -> int:
    n = int(max(1, n_total))
    if min_expected <= 0:
        return int(max(2, Q_min))
    Q = int(math.floor(math.sqrt(n / float(min_expected))))
    Q = max(int(Q_min), min(int(Q_max), max(2, Q)))
    return int(Q)


def _resolve_sampling_rate(*, policy: BinsPolicy) -> Tuple[Optional[float], str]:
    fs = policy.spectral.fs_hz
    if fs is None:
        return None, "cycles_per_sample"
    fs = float(fs)
    if fs <= 0:
        raise InformationalBinsError(f"Invalid spectral fs_hz: {fs}")
    return fs, "Hz"


def _window_values(name: str, n: int) -> np.ndarray:
    name = str(name).strip().lower()
    if name in ("hann", "hanning"):
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name in ("boxcar", "rect", "rectangular", "none"):
        return np.ones(n, dtype=float)
    raise InformationalBinsError(f"Unsupported window: {name!r}")


def _build_spectral_edges_equal_energy(
    *,
    parquet_paths: Sequence[Path],
    laser_col: str,
    fs: float,
    n_fft: int,
    blocks_per_file: int,
    max_files: int,
    window: str,
    K_min: int,
    K_max: int,
    epsilon: float,
    rng: np.random.Generator,
    parquet_engine: str,
) -> Tuple[np.ndarray, int]:
    """
    Build spectral frequency edges such that average control PSD energy is roughly equal per band.
    """
    fs = float(fs)
    if fs <= 0:
        raise InformationalBinsError("fs must be > 0 for spectral edges.")
    n_fft = int(n_fft)
    if n_fft < 64:
        raise InformationalBinsError("n_fft too small for spectral edges.")
    blocks_per_file = int(max(1, blocks_per_file))
    max_files = int(max(1, max_files))

    paths = list(parquet_paths)[:max_files]
    if not paths:
        raise InformationalBinsError("No reference parquets available for spectral edges.")

    win = _window_values(window, n_fft).astype(float)
    win_norm = float(np.sum(win * win))
    if win_norm <= 0:
        raise InformationalBinsError("Invalid window normalization.")

    R = n_fft // 2 + 1
    psd_sum = np.zeros(R, dtype=float)
    n_blocks = 0

    for p in paths:
        x = _read_parquet_column(p, laser_col, engine=parquet_engine)
        x = x[np.isfinite(x)]
        if x.size < n_fft:
            continue

        max_start = x.size - n_fft
        n_take = min(blocks_per_file, max(1, (max_start + 1)))

        if max_start < blocks_per_file:
            starts = np.linspace(0, max_start, n_take, dtype=int)
        else:
            starts = rng.integers(0, max_start + 1, size=n_take, endpoint=False)

        for s in starts:
            seg = x[int(s): int(s) + n_fft]
            if seg.size != n_fft or not np.all(np.isfinite(seg)):
                continue
            seg = seg - float(np.mean(seg))
            Y = np.fft.rfft(seg * win)
            P = (Y.real * Y.real + Y.imag * Y.imag) / win_norm
            psd_sum += P
            n_blocks += 1

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    if n_blocks < 2:
        K = int(max(K_min, min(K_max, int(round(math.sqrt(R))))))
        K = max(2, min(K, R - 1))
        edges = np.linspace(float(freqs[0]), float(freqs[-1]), K + 1, dtype=float)
        _enforce_strictly_increasing(edges, float(epsilon))
        return edges, int(K)

    psd_avg = psd_sum / float(n_blocks)
    psd_avg = np.maximum(psd_avg, 0.0)

    total = float(np.sum(psd_avg))
    if not (np.isfinite(total) and total > 0):
        raise InformationalBinsError(f"Invalid average PSD energy for {laser_col}.")

    cdf = np.cumsum(psd_avg) / total  # in [0,1]

    K = int(round(math.sqrt(R)))
    K = int(max(K_min, min(K_max, max(2, K))))
    K = min(K, R - 1)

    targets = np.linspace(0.0, 1.0, K + 1, dtype=float)
    idx = np.searchsorted(cdf, targets, side="left")
    idx = np.clip(idx, 0, freqs.size - 1)

    edges = freqs[idx].astype(float)
    edges[0] = float(freqs[0])
    edges[-1] = float(freqs[-1])

    _enforce_strictly_increasing(edges, float(epsilon))
    return edges, int(K)


def _enforce_strictly_increasing(arr: np.ndarray, epsilon: float) -> None:
    eps = float(epsilon)
    for i in range(1, arr.size):
        if arr[i] <= arr[i - 1]:
            arr[i] = arr[i - 1] + eps


__all__ = [
    "InformationalBinsError",
    "QuantileEdgesPolicy",
    "IncrementsPolicy",
    "CouplingPolicy",
    "SpectralPolicy",
    "BinsPolicy",
    "BinsArtifacts",
    "build_bins_spec_from_level4_config_json",
    "build_bins_spec_from_config",
    "build_bins_spec",
]
