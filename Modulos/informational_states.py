# informational_states.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import math
import numpy as np
import pandas as pd


# =============================================================================
# Exceptions
# =============================================================================

class InformationalStatesError(RuntimeError):
    """Raised when informational states (PMFs) cannot be constructed."""


# =============================================================================
# Policy / Artifacts
# =============================================================================

@dataclass(frozen=True)
class StatesPolicy:
    # Core math
    log_base: float = 2.0
    remove_dc: bool = True

    # Parquet IO
    parquet_engine: str = "auto"  # "auto" uses pandas default
    prefer_column_read: bool = True  # try to read only needed columns

    # Optional fs estimation (only needed if spectral unit is Hz and fs missing)
    estimate_fs_if_missing: bool = False
    time_col_candidates: Tuple[str, ...] = ("t", "time", "timestamp", "Time", "Samples")

    # Controls
    require_spectral: bool = False
    coupling_required: bool = False

    # Streaming output (low-resource)
    pmf_batch_rows: int = 250_000  # flush PMF rows when batch reaches this size
    max_rows_per_parquet: Optional[int] = None  # cap rows per measurement (debug / low-resource)

    # Numerical
    eps: float = 1e-12


@dataclass(frozen=True)
class InformationalStatesArtifacts:
    states_summary_csv: Path
    pmf_long_path: Path
    coupling_mi_csv: Path


# =============================================================================
# Public API
# =============================================================================

def build_states_from_bins(
    *,
    bins_spec_json: Union[str, Path],
    quality_queue_csv: Union[str, Path],
    output_dir: Union[str, Path],
    policy: StatesPolicy = StatesPolicy(),
) -> InformationalStatesArtifacts:
    """
    Build informational states (PMFs + entropies) per measurement using frozen bins.

    Inputs
    ------
    bins_spec_json:
        Level2 bins_spec.json (schema_version=2.0 preferred; legacy tolerated as fallback).
    quality_queue_csv:
        Queue CSV with at least: mid, parquet_path.
    output_dir:
        Folder where artifacts will be written.

    Outputs
    -------
    - states_summary.csv: one row per (mid, laser)
    - pmf_long.parquet (or .csv fallback): long-format PMFs
    - coupling_mi.csv: MI per pair per mid (optional; written if coupling edges exist)
    """
    bins_spec_json = Path(bins_spec_json)
    quality_queue_csv = Path(quality_queue_csv)
    output_dir = Path(output_dir)

    _require_file(bins_spec_json, "bins_spec_json")
    _require_file(quality_queue_csv, "quality_queue_csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _read_json(bins_spec_json)
    
    
    schema_version = str(spec.get("schema_version", ""))
    is_v2 = schema_version == "2.0" and isinstance(spec.get("channels"), list)

    # Parse bins (v2 primary; legacy fallback)
    if is_v2:
        bins_by_laser = _parse_bins_spec_channels_v2(spec)
        spec_lasers = [str(x) for x in (spec.get("lasers") or [])] or list(bins_by_laser.keys())
    else:
        bins_by_laser, spec_lasers = _parse_bins_spec_legacy(spec)

    if not spec_lasers:
        raise InformationalStatesError("No lasers found in bins_spec.")


    # Load queue
    qdf = _load_queue_csv(quality_queue_csv)

    # Outputs
    states_summary_csv = output_dir / "states_summary.csv"
    coupling_mi_csv = output_dir / "coupling_mi.csv"

    # PMF output: try parquet; if unavailable, fall back to csv
    pmf_long_parquet = output_dir / "pmf_long.parquet"
    pmf_long_csv = output_dir / "pmf_long.csv"
    pmf_writer = _PMFWriter(pmf_long_parquet, pmf_long_csv)

    summary_rows: List[Dict[str, Any]] = []
    coupling_rows: List[Dict[str, Any]] = []

    for _, qrow in qdf.iterrows():
        mid = str(qrow.get("mid", "")).strip()
        parquet_path = Path(str(qrow.get("parquet_path", "")).strip())

        if not mid:
            summary_rows.append({"mid": "", "error": "Missing mid in queue."})
            continue
        if not parquet_path.exists():
            summary_rows.append({"mid": mid, "error": f"Missing parquet: {parquet_path}"})
            continue

        # Determine lasers to use for this measurement from bins_spec only
        lasers = [l for l in spec_lasers if l in bins_by_laser]
        if not lasers:
            summary_rows.append({"mid": mid, "error": "No lasers available from bins_spec."})
            continue

        # Read parquet columns (lasers + optional time column if needed)
        time_col = None
        needs_time = policy.estimate_fs_if_missing
        parquet_cols = _select_parquet_columns(
            parquet_path=parquet_path,
            lasers=lasers,
            time_col_candidates=policy.time_col_candidates,
            need_time=needs_time,
        )
        if parquet_cols is None:
            # fallback (read full)
            df = _read_parquet_any(parquet_path, engine=policy.parquet_engine)
        else:
            df = _read_parquet_cols(
                parquet_path, cols=parquet_cols, engine=policy.parquet_engine, prefer_column_read=policy.prefer_column_read
            )

        # Identify time column if present
        for c in policy.time_col_candidates:
            if c in df.columns:
                time_col = c
                break

        # Filter lasers to those present
        lasers = [l for l in lasers if l in df.columns]
        if not lasers:
            summary_rows.append({"mid": mid, "error": "No laser columns found in parquet."})
            continue

        X = df[lasers].to_numpy(dtype=float, copy=False)
        if policy.max_rows_per_parquet is not None and X.shape[0] > int(policy.max_rows_per_parquet):
            X = X[: int(policy.max_rows_per_parquet), :]

        N, d = X.shape
        base_meta = _extract_meta(qrow)
        base_meta.update(
            {
                "mid": mid,
                "parquet_path": str(parquet_path),
                "N": int(N),
                "d": int(d),
                "bins_mode": str(spec.get("mode", "")),
                "reference_group": str((spec.get("reference") or {}).get("group", "")),
                "bins_spec_path": str(bins_spec_json),
            }
        )

        # Resolve per-laser bin contracts in the same order as `lasers`
        amp_edges_list: List[np.ndarray] = []
        inc_edges_list: List[np.ndarray] = []
        coup_edges_list: List[Optional[np.ndarray]] = []
        spec_edges_list: List[Optional[np.ndarray]] = []
        spec_fs_list: List[Optional[float]] = []
        spec_unit_list: List[Optional[str]] = []
        amp_clip_list: List[bool] = []
        inc_clip_list: List[bool] = []
        coup_clip_list: List[bool] = []

        for l in lasers:
            entry = bins_by_laser.get(l)
            if entry is None:
                raise InformationalStatesError(f"bins_spec: missing entry for laser '{l}'.")
            amp_edges_list.append(np.asarray(entry["amplitude_edges"], dtype=float))
            inc_edges_list.append(np.asarray(entry["increment_edges"], dtype=float))
            coup_edges_list.append(np.asarray(entry["coupling_edges"], dtype=float) if entry.get("coupling_edges") is not None else None)
            spec_edges_list.append(np.asarray(entry["spectral_freq_edges"], dtype=float) if entry.get("spectral_freq_edges") is not None else None)
            spec_fs_list.append(entry.get("spectral_fs"))
            spec_unit_list.append(entry.get("spectral_unit"))
            amp_clip_list.append(bool(entry.get("amp_clip", True)))
            inc_clip_list.append(bool(entry.get("inc_clip", True)))
            coup_clip_list.append(bool(entry.get("coup_clip", True)))

        # -----------------------------------------------------------------------------
        # 1) Amplitude state
        # -----------------------------------------------------------------------------
        amp_pmf_list, H_amp = amplitude_state_by_edges(
            X,
            edges_by_channel=amp_edges_list,
            clip_by_channel=amp_clip_list,
            log_base=policy.log_base,
            eps=policy.eps,
        )

        # -----------------------------------------------------------------------------
        # 2) Movement (increments) state
        # -----------------------------------------------------------------------------
        mov_pmf_list, H_mov = movement_state_by_edges(
            X,
            edges_by_channel=inc_edges_list,
            clip_by_channel=inc_clip_list,
            log_base=policy.log_base,
            eps=policy.eps,
        )

        # -----------------------------------------------------------------------------
        # 3) Energy and 4) Fourier (optional; per channel)
        # -----------------------------------------------------------------------------
        H_energy = np.full(d, np.nan, dtype=float)
        H_fourier = np.full(d, np.nan, dtype=float)
        energy_pmf_list: List[Optional[np.ndarray]] = [None] * d
        fourier_pmf_list: List[Optional[np.ndarray]] = [None] * d

        for i in range(d):
            freq_edges = spec_edges_list[i]
            if freq_edges is None:
                if policy.require_spectral:
                    raise InformationalStatesError(f"Spectral bins missing for laser '{lasers[i]}' and require_spectral=True.")
                continue

            unit = (spec_unit_list[i] or "").lower().strip()
            fs = spec_fs_list[i]
            if fs is None or (isinstance(fs, (int, float)) and float(fs) <= 0):
                # If unit is unitless (cycles_per_sample), default to 1.0
                if unit in ("cycles_per_sample", "cps", "unitless", ""):
                    fs_eff = 1.0
                else:
                    # Hz requires fs
                    if policy.estimate_fs_if_missing and time_col is not None:
                        fs_eff = _estimate_fs_from_time(df[time_col].to_numpy(dtype=float, copy=False))
                    else:
                        raise InformationalStatesError(
                            f"fs missing for spectral unit '{unit}' (laser={lasers[i]}, mid={mid})."
                        )
            else:
                fs_eff = float(fs)

            e_pmf, eH = energy_state_by_edges(
                X[:, i],
                fs=fs_eff,
                freq_edges=freq_edges,
                log_base=policy.log_base,
                remove_dc=policy.remove_dc,
                eps=policy.eps,
            )
            f_pmf, fH = fourier_state_by_edges(
                X[:, i],
                fs=fs_eff,
                freq_edges=freq_edges,
                log_base=policy.log_base,
                remove_dc=policy.remove_dc,
                eps=policy.eps,
            )

            energy_pmf_list[i] = e_pmf
            fourier_pmf_list[i] = f_pmf
            H_energy[i] = float(eH)
            H_fourier[i] = float(fH)

        # -----------------------------------------------------------------------------
        # 5) Coupling MI (optional)
        # -----------------------------------------------------------------------------
        has_coupling = all(c is not None for c in coup_edges_list)
        if has_coupling:
            edges_for_mi = [np.asarray(c, dtype=float) for c in coup_edges_list]  # type: ignore[arg-type]
            mi = coupling_mutual_information_by_edges(X, edges_by_channel=edges_for_mi, log_base=policy.log_base)
            mi_mat = mi["MI_matrix"]

            for i in range(d):
                for j in range(i + 1, d):
                    coupling_rows.append(
                        {
                            **base_meta,
                            "laser_i": lasers[i],
                            "laser_j": lasers[j],
                            "MI": float(mi_mat[i, j]),
                        }
                    )
        elif policy.coupling_required:
            raise InformationalStatesError("Coupling edges missing and coupling_required=True.")

        # -----------------------------------------------------------------------------
        # Summary rows: one per (mid, laser)
        # -----------------------------------------------------------------------------
        for i, l in enumerate(lasers):
            summary_rows.append(
                {
                    **base_meta,
                    "laser": l,
                    "H_amp": float(H_amp[i]),
                    "H_mov": float(H_mov[i]),
                    "H_energy": float(H_energy[i]) if np.isfinite(H_energy[i]) else float("nan"),
                    "H_fourier": float(H_fourier[i]) if np.isfinite(H_fourier[i]) else float("nan"),
                }
            )

        # -----------------------------------------------------------------------------
        # PMF long rows (streamed)
        # -----------------------------------------------------------------------------
        pmf_writer.add_rows(_pmf_to_long_rows(mid, base_meta, lasers, "amplitude", amp_pmf_list, edges_by_laser=_edges_map(lasers, amp_edges_list)))
        pmf_writer.add_rows(_pmf_to_long_rows(mid, base_meta, lasers, "movement", mov_pmf_list, edges_by_laser=_edges_map(lasers, inc_edges_list)))

        # Spectral PMFs are per channel; keep feature bins consistent with freq_edges
        for i, l in enumerate(lasers):
            if energy_pmf_list[i] is not None:
                pmf_writer.add_rows(_pmf_to_long_rows(mid, base_meta, [l], "energy", [energy_pmf_list[i]], edges_by_laser=None))
            if fourier_pmf_list[i] is not None:
                pmf_writer.add_rows(_pmf_to_long_rows(mid, base_meta, [l], "fourier_binned", [fourier_pmf_list[i]], edges_by_laser=None))

        pmf_writer.maybe_flush(policy.pmf_batch_rows)

    # Final flush
    pmf_writer.flush()
    pmf_writer.close()

    # Write CSV outputs (summary + coupling)
    pd.DataFrame(summary_rows).to_csv(states_summary_csv, index=False, encoding="utf-8")
    pd.DataFrame(coupling_rows).to_csv(coupling_mi_csv, index=False, encoding="utf-8")

    # Return artifacts
    pmf_path = pmf_writer.final_path
    return InformationalStatesArtifacts(
        states_summary_csv=states_summary_csv,
        pmf_long_path=pmf_path,
        coupling_mi_csv=coupling_mi_csv,
    )


# =============================================================================
# States (PMFs)
# =============================================================================

def amplitude_state_by_edges(
    X: np.ndarray,
    *,
    edges_by_channel: Sequence[np.ndarray],
    clip_by_channel: Sequence[bool],
    log_base: float = 2.0,
    eps: float = 1e-12,
) -> Tuple[List[np.ndarray], np.ndarray]:
    X = _ensure_2d(X)
    N, d = X.shape
    if len(edges_by_channel) != d or len(clip_by_channel) != d:
        raise ValueError("edges_by_channel and clip_by_channel must match X.shape[1].")

    pmfs: List[np.ndarray] = []
    H = np.zeros(d, dtype=float)
    for i in range(d):
        edges = np.asarray(edges_by_channel[i], dtype=float)
        B = int(edges.size - 1)
        if B <= 0:
            raise ValueError("Invalid edges for amplitude state.")
        x = X[:, i]
        if bool(clip_by_channel[i]):
            x = np.clip(x, float(edges[0]), float(edges[-1]))
        idx = np.searchsorted(edges, x, side="right") - 1
        idx = np.clip(idx, 0, B - 1).astype(np.int32, copy=False)
        counts = np.bincount(idx, minlength=B).astype(float)
        p = counts / float(max(N, 1))
        pmfs.append(p)
        H[i] = _entropy_from_pmf(p, log_base=log_base, eps=eps)
    return pmfs, H


def movement_state_by_edges(
    X: np.ndarray,
    *,
    edges_by_channel: Sequence[np.ndarray],
    clip_by_channel: Sequence[bool],
    log_base: float = 2.0,
    eps: float = 1e-12,
) -> Tuple[List[np.ndarray], np.ndarray]:
    X = _ensure_2d(X)
    N, d = X.shape
    if len(edges_by_channel) != d or len(clip_by_channel) != d:
        raise ValueError("edges_by_channel and clip_by_channel must match X.shape[1].")

    pmfs: List[np.ndarray] = []
    H = np.zeros(d, dtype=float)
    for i in range(d):
        edges = np.asarray(edges_by_channel[i], dtype=float)
        B = int(edges.size - 1)
        if B <= 0:
            raise ValueError("Invalid edges for movement state.")
        x = X[:, i]
        if x.size < 2:
            p = np.zeros(B, dtype=float)
            p[0] = 1.0
            pmfs.append(p)
            H[i] = 0.0
            continue
        dx = np.diff(x)
        if bool(clip_by_channel[i]):
            dx = np.clip(dx, float(edges[0]), float(edges[-1]))
        idx = np.searchsorted(edges, dx, side="right") - 1
        idx = np.clip(idx, 0, B - 1).astype(np.int32, copy=False)
        counts = np.bincount(idx, minlength=B).astype(float)
        p = counts / float(max(dx.size, 1))
        pmfs.append(p)
        H[i] = _entropy_from_pmf(p, log_base=log_base, eps=eps)
    return pmfs, H


def energy_state_by_edges(
    x: np.ndarray,
    *,
    fs: float,
    freq_edges: np.ndarray,
    log_base: float = 2.0,
    remove_dc: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float).reshape(-1,)
    N = int(x.size)
    edges = np.asarray(freq_edges, dtype=float)
    K = int(edges.size - 1)
    if N <= 0 or K <= 0:
        return np.zeros(max(K, 0), dtype=float), 0.0

    y = x.copy()
    if remove_dc:
        y -= float(np.mean(y))

    fs = float(fs)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    Y = np.fft.rfft(y)
    power = (Y.real * Y.real + Y.imag * Y.imag)

    band_idx = np.searchsorted(edges, freqs, side="right") - 1
    mask = (band_idx >= 0) & (band_idx < K)
    E = np.bincount(band_idx[mask], weights=power[mask], minlength=K).astype(float)

    p = _safe_normalize(E, eps=eps)
    return p, float(_entropy_from_pmf(p, log_base=log_base, eps=eps))


def fourier_state_by_edges(
    x: np.ndarray,
    *,
    fs: float,
    freq_edges: np.ndarray,
    log_base: float = 2.0,
    remove_dc: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float).reshape(-1,)
    N = int(x.size)
    edges = np.asarray(freq_edges, dtype=float)
    K = int(edges.size - 1)
    if N <= 0 or K <= 0:
        return np.zeros(max(K, 0), dtype=float), 0.0

    y = x.copy()
    if remove_dc:
        y -= float(np.mean(y))

    fs = float(fs)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    Y = np.fft.rfft(y)
    power = (Y.real * Y.real + Y.imag * Y.imag)

    pi = _safe_normalize(power, eps=eps)

    band_idx = np.searchsorted(edges, freqs, side="right") - 1
    mask = (band_idx >= 0) & (band_idx < K)
    Pb = np.bincount(band_idx[mask], weights=pi[mask], minlength=K).astype(float)
    pb = _safe_normalize(Pb, eps=eps)
    Hbinned = _entropy_from_pmf(pb, log_base=log_base, eps=eps)

    return pb, float(Hbinned)

def coupling_mutual_information_by_edges(
    X: np.ndarray,
    *,
    edges_by_channel: Sequence[np.ndarray],
    log_base: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Mutual information per pair using per-channel discretization edges.
    Supports different Q per channel. Uses vectorized bincount.
    """
    X = _ensure_2d(X)
    N, d = X.shape
    if len(edges_by_channel) != d:
        raise ValueError("edges_by_channel must have length equal to number of columns in X.")

    ln_base = math.log(float(log_base))
    cats: List[np.ndarray] = []
    Qs: List[int] = []

    for i in range(d):
        edges = np.asarray(edges_by_channel[i], dtype=float)
        Qi = int(edges.size - 1)
        if Qi <= 0:
            raise ValueError("Each channel must have >= 1 coupling bin.")
        idx = np.searchsorted(edges, X[:, i], side="right") - 1
        ci = np.clip(idx, 0, Qi - 1).astype(np.int32, copy=False)
        cats.append(ci)
        Qs.append(Qi)

    MI = np.zeros((d, d), dtype=float)
    if N <= 0:
        return {"MI_matrix": MI, "MI_vector": np.zeros(0, dtype=float)}

    for i in range(d):
        ai = cats[i]
        Qi = Qs[i]
        for j in range(i + 1, d):
            aj = cats[j]
            Qj = Qs[j]
            flat = ai * Qj + aj
            counts = np.bincount(flat, minlength=Qi * Qj).astype(float)
            p_ij = counts.reshape(Qi, Qj) / float(N)

            p_i = np.sum(p_ij, axis=1, keepdims=True)
            p_j = np.sum(p_ij, axis=0, keepdims=True)
            denom = p_i @ p_j

            mask = p_ij > 0
            if not np.any(mask):
                Iij = 0.0
            else:
                ratio = p_ij[mask] / denom[mask]
                Iij = float(np.sum(p_ij[mask] * (np.log(ratio) / ln_base)))

            MI[i, j] = Iij
            MI[j, i] = Iij

    vec = []
    for i in range(d):
        for j in range(i + 1, d):
            vec.append(MI[i, j])

    return {"MI_matrix": MI, "MI_vector": np.asarray(vec, dtype=float)}


# =============================================================================
# PMF output helpers (streamed)
# =============================================================================

class _PMFWriter:
    """
    Stream PMF rows to parquet if available; otherwise to CSV.

    This avoids accumulating millions of rows in RAM.
    """

    def __init__(self, parquet_path: Path, csv_path: Path):
        self.parquet_path = Path(parquet_path)
        self.csv_path = Path(csv_path)
        self._rows: List[Dict[str, Any]] = []
        self._use_parquet: Optional[bool] = None
        self._parquet_writer = None  # pyarrow.parquet.ParquetWriter
        self._schema = None
        self.final_path: Path = self.parquet_path  # may switch to csv

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        self._rows.extend(rows)

    def maybe_flush(self, threshold_rows: int) -> None:
        if len(self._rows) >= int(threshold_rows):
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return

        df = pd.DataFrame(self._rows)
        self._rows = []

        if self._use_parquet is None:
            self._use_parquet = _can_write_parquet()

        if self._use_parquet:
            try:
                self._append_parquet(df)
                self.final_path = self.parquet_path
                return
            except Exception:
                # Fall back to CSV; keep going
                self._use_parquet = False

        self._append_csv(df)
        self.final_path = self.csv_path

    def _append_csv(self, df: pd.DataFrame) -> None:
        header = not self.csv_path.exists()
        df.to_csv(self.csv_path, mode="a", index=False, header=header, encoding="utf-8")

    def _append_parquet(self, df: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._parquet_writer is None:
            self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
            self._schema = table.schema
            self._parquet_writer = pq.ParquetWriter(self.parquet_path, self._schema, compression="snappy")
        self._parquet_writer.write_table(table)

    def close(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _pmf_to_long_rows(
    mid: str,
    meta: Dict[str, Any],
    lasers: Sequence[str],
    feature: str,
    pmf_list: Sequence[np.ndarray],
    *,
    edges_by_laser: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, l in enumerate(lasers):
        p = np.asarray(pmf_list[i], dtype=float).reshape(-1,)
        for b in range(p.size):
            rec: Dict[str, Any] = {
                **meta,
                "mid": mid,
                "laser": str(l),
                "feature": str(feature),
                "bin": int(b),
                "prob": float(p[b]),
            }
            if edges_by_laser is not None and l in edges_by_laser:
                e = edges_by_laser[l]
                if 0 <= b < (len(e) - 1):
                    rec["bin_left"] = float(e[b])
                    rec["bin_right"] = float(e[b + 1])
            rows.append(rec)
    return rows


# =============================================================================
# Bins spec parsing
# =============================================================================

def _parse_bins_spec_channels_v2(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Primary parser for bins_spec schema_version=2.0 using spec['channels'].

    Returns:
      laser -> {
        amplitude_edges: np.ndarray,
        increment_edges: np.ndarray,
        coupling_edges: Optional[np.ndarray],
        spectral_freq_edges: Optional[np.ndarray],
        spectral_fs: Optional[float],
        spectral_unit: Optional[str],
        amp_clip/inc_clip/coup_clip: bool
      }
    """
    ch_list = spec.get("channels")
    if not isinstance(ch_list, list) or not ch_list:
        raise InformationalStatesError("bins_spec v2 requires a non-empty 'channels' list.")

    legacy_clip: Dict[str, Dict[str, bool]] = {}
    if isinstance(spec.get("bins"), list):
        for item in spec["bins"]:
            laser = str(item.get("laser", "")).strip()
            if not laser:
                continue
            legacy_clip[laser] = {
                "amp_clip": bool((item.get("amplitude") or {}).get("clip_out_of_range", True)),
                "inc_clip": bool((item.get("increments") or {}).get("clip_out_of_range", True)),
                "coup_clip": bool((item.get("coupling_categories") or {}).get("clip_out_of_range", True)),
            }

    # Optional global spectral (fallback)
    global_spectral = spec.get("spectral", None) or {}
    global_freq_edges = global_spectral.get("freq_edges", None)
    global_unit = global_spectral.get("freq_unit", None)
    global_fs = global_spectral.get("fs", None)

    out: Dict[str, Dict[str, Any]] = {}
    for item in ch_list:
        if not isinstance(item, dict) or "laser" not in item:
            continue
        laser = str(item["laser"]).strip()
        if not laser:
            continue

        a = np.asarray(item.get("amplitude_edges"), dtype=float)
        dx = np.asarray(item.get("increment_edges"), dtype=float)
        _ensure_edges_ok(a, ctx=f"{laser}:amplitude_edges")
        _ensure_edges_ok(dx, ctx=f"{laser}:increment_edges")

        coup = item.get("coupling_edges", None)
        coup_arr = np.asarray(coup, dtype=float) if coup is not None else None
        if coup_arr is not None:
            _ensure_edges_ok(coup_arr, ctx=f"{laser}:coupling_edges")

        spectral_node = item.get("spectral", None) or {}
        freq_edges = spectral_node.get("freq_edges", global_freq_edges)
        freq_arr = np.asarray(freq_edges, dtype=float) if freq_edges is not None else None
        if freq_arr is not None:
            _ensure_edges_ok(freq_arr, ctx=f"{laser}:spectral.freq_edges")

        unit = spectral_node.get("freq_unit", global_unit)
        fs = spectral_node.get("fs", global_fs)
        fs_val = float(fs) if fs is not None else None

        clips = legacy_clip.get(laser, {"amp_clip": True, "inc_clip": True, "coup_clip": True})

        out[laser] = {
            "amplitude_edges": a,
            "increment_edges": dx,
            "coupling_edges": coup_arr,
            "spectral_freq_edges": freq_arr,
            "spectral_fs": fs_val,
            "spectral_unit": unit,
            "amp_clip": bool(clips["amp_clip"]),
            "inc_clip": bool(clips["inc_clip"]),
            "coup_clip": bool(clips["coup_clip"]),
        }

    if not out:
        raise InformationalStatesError("bins_spec v2 parsed no valid channel entries.")
    return out


def _parse_bins_spec_legacy(spec: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Legacy fallback parser.

    Expected (legacy) keys (best-effort):
      spec['bins'] list with:
        - laser
        - amplitude.edges
        - increments.edges
        - coupling_categories.edges (optional)
        - spectral.freq_edges or energy_band_edges_hz/fourier_edges_hz (optional)

    This is maintained only to avoid breaking old experiments.
    """
    bins_list = spec.get("bins")
    if not isinstance(bins_list, list) or not bins_list:
        raise InformationalStatesError("Legacy bins_spec requires non-empty 'bins' list.")

    out: Dict[str, Dict[str, Any]] = {}
    lasers: List[str] = []

    for b in bins_list:
        if not isinstance(b, dict):
            continue
        laser = str(b.get("laser", "")).strip()
        if not laser:
            continue

        amp = (b.get("amplitude") or {})
        inc = (b.get("increments") or {})
        aedges = np.asarray(amp.get("edges"), dtype=float)
        iedges = np.asarray(inc.get("edges"), dtype=float)
        _ensure_edges_ok(aedges, ctx=f"{laser}:amplitude.edges")
        _ensure_edges_ok(iedges, ctx=f"{laser}:increments.edges")

        coup = (b.get("coupling_categories") or {}).get("edges", None)
        coup_arr = np.asarray(coup, dtype=float) if coup is not None else None
        if coup_arr is not None:
            _ensure_edges_ok(coup_arr, ctx=f"{laser}:coupling.edges")

        # Spectral (legacy variants)
        spectral_node = b.get("spectral", None) or spec.get("spectral", None) or {}
        freq_edges = spectral_node.get("freq_edges", None)

        freq_arr = np.asarray(freq_edges, dtype=float) if freq_edges is not None else None
        if freq_arr is not None:
            _ensure_edges_ok(freq_arr, ctx=f"{laser}:spectral.freq_edges")

        unit = spectral_node.get("freq_unit", spectral_node.get("freq_unit", None))
        fs = spectral_node.get("fs", spectral_node.get("fs_hz", None))
        fs_val = float(fs) if fs is not None else None

        out[laser] = {
            "amplitude_edges": aedges,
            "increment_edges": iedges,
            "coupling_edges": coup_arr,
            "spectral_freq_edges": freq_arr,
            "spectral_fs": fs_val,
            "spectral_unit": unit,
            "amp_clip": bool(amp.get("clip_out_of_range", True)),
            "inc_clip": bool(inc.get("clip_out_of_range", True)),
            "coup_clip": bool((b.get("coupling_categories") or {}).get("clip_out_of_range", True)),
        }
        lasers.append(laser)

    if not out:
        raise InformationalStatesError("Legacy bins_spec parsed no valid entries.")
    lasers = sorted(set(lasers))
    return out, lasers


# =============================================================================
# IO helpers
# =============================================================================

def _require_file(path: Path, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        raise InformationalStatesError(f"Failed to read JSON: {path} ({type(e).__name__}: {e})") from e


def _load_queue_csv(path: Path) -> pd.DataFrame:
    df = _read_csv_robust(path)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    # Minimal contract
    if "parquet_path" not in df.columns:
        raise InformationalStatesError("Queue CSV missing required column: parquet_path")
    if "mid" not in df.columns:
        # Best-effort patch: if archivo exists, use it as mid; otherwise fail
        if "archivo" in df.columns:
            df["mid"] = df["archivo"].astype(str)
        else:
            raise InformationalStatesError("Queue CSV missing required column: mid")

    # Keep stable ordering (as written by Level1)
    return df




def _read_csv_robust(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise InformationalStatesError(f"Failed to read CSV: {path} ({type(last_err).__name__}: {last_err})")


def _read_parquet_any(path: Path, engine: str = "auto") -> pd.DataFrame:
    try:
        if engine == "auto":
            return pd.read_parquet(path)
        return pd.read_parquet(path, engine=engine)
    except Exception as e:
        raise InformationalStatesError(f"Failed reading parquet: {path} ({type(e).__name__}: {e})") from e


def _read_parquet_cols(
    path: Path,
    *,
    cols: Sequence[str],
    engine: str = "auto",
    prefer_column_read: bool = True,
) -> pd.DataFrame:
    cols = [str(c) for c in cols]
    if not prefer_column_read:
        df = _read_parquet_any(path, engine=engine)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise InformationalStatesError(f"Parquet missing columns: {missing} ({path})")
        return df[cols].copy()

    try:
        if engine == "auto":
            return pd.read_parquet(path, columns=list(cols))
        return pd.read_parquet(path, columns=list(cols), engine=engine)
    except Exception:
        # Fallback: read full and subset
        df = _read_parquet_any(path, engine=engine)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise InformationalStatesError(f"Parquet missing columns: {missing} ({path})")
        return df[cols].copy()


def _select_parquet_columns(
    *,
    parquet_path: Path,
    lasers: Sequence[str],
    time_col_candidates: Sequence[str],
    need_time: bool,
) -> Optional[List[str]]:
    """
    Try to select only necessary parquet columns without loading full data.
    If pyarrow is available, we can inspect schema cheaply and return a safe subset.
    Returns None if schema inspection is not available.
    """
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        names = set(pf.schema.names)
        cols: List[str] = [l for l in lasers if l in names]
        if need_time:
            for c in time_col_candidates:
                if c in names:
                    cols.append(c)
                    break
        return cols if cols else None
    except Exception:
        return None


# =============================================================================
# Math helpers
# =============================================================================

def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 1D or 2D array.")
    return X


def _safe_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = float(np.sum(x))
    if not np.isfinite(s) or s <= eps:
        out = np.zeros_like(x, dtype=float)
        if out.size > 0:
            out[0] = 1.0
        return out
    return x / s


def _entropy_from_pmf(p: np.ndarray, *, log_base: float = 2.0, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float).reshape(-1,)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    denom = math.log(float(log_base))
    return float(-np.sum(p * (np.log(p + eps) / denom)))


def _ensure_edges_ok(edges: np.ndarray, *, ctx: str) -> None:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise InformationalStatesError(f"Invalid edges shape for {ctx}: {edges.shape}")
    if not np.all(np.isfinite(edges)):
        raise InformationalStatesError(f"Non-finite edges for {ctx}")
    if not np.all(edges[1:] > edges[:-1]):
        raise InformationalStatesError(f"Edges not strictly increasing for {ctx}")


def _estimate_fs_from_time(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float).reshape(-1,)
    if t.size < 3:
        return float("nan")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return float("nan")
    med = float(np.median(dt))
    if med <= 0:
        return float("nan")
    return float(1.0 / med)


# =============================================================================
# Misc
# =============================================================================

def _extract_meta(qrow: pd.Series) -> Dict[str, Any]:
    # Keep only active / tolerated non-structural metadata
    keys = ("fecha", "lab", "archivo", "color")
    out = {}
    for k in keys:
        if k in qrow.index:
            out[k] = qrow.get(k)
    return out

def _edges_map(lasers: Sequence[str], edges_list: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    return {str(lasers[i]): np.asarray(edges_list[i], dtype=float) for i in range(len(lasers))}


def _can_write_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False
