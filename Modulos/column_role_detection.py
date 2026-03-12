# column_role_detection.py
"""
Column Role Detection (part 1/2)

Goal
----
From a dialect-enriched analysis catalog (DataFrame), detect per CSV:
- t_sys: system time column (HH:MM:SS or HH:MM:SS.mmm)
- t_rel: relative time column (HH:MM:SS... typically starts near 0 and increases)
- channels: numeric signal columns

This module is:
- read-only (does NOT rewrite CSVs)
- experimental-mode agnostic (blind/declared does not matter here)
- low-resource friendly (streaming read, small row sampling)

Inputs
------
- catalog_df (pandas.DataFrame), typically output of CSVDialectInspector.run(...)
  Required columns: mid, raw_path, raw_exists
  Optional columns: csv_extra_field_candidate, csv_extra_field_constant_value,
                    csv_delimiter, csv_encoding, csv_dialect_ok

Outputs
-------
- column_roles.json (written to cfg.target_root by default)

Notes
-----
- Extra-field handling: if csv_extra_field_candidate is True, we drop a trailing field
  ONLY when a row has exactly (header_ncols + 1) fields. This is safe and avoids
  accidentally truncating valid files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import json
import math

import pandas as pd

from config import ExperimentConfig


# -----------------------------
# Public configuration
# -----------------------------
@dataclass(frozen=True)
class ColumnRoleSpec:
    time_system_candidates: List[str]
    time_relative_candidates: List[str]
    channel_name_pool: List[str]

    detect_rows: int = 80
    min_parse_ratio_time: float = 0.80
    min_parse_ratio_numeric: float = 0.80
    require_two_time_cols: bool = True


@dataclass(frozen=True)
class ColumnRoleArtifacts:
    role_map_path: Path


# -----------------------------
# Helpers (small, focused)
# -----------------------------
_DEFAULT_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")


def _norm_name(s: str) -> str:
    s2 = (s or "").strip().lower().replace(" ", "_")
    return "".join(ch for ch in s2 if ch.isalnum() or ch == "_")


def _open_text(path: Path, preferred_encoding: Optional[str]) -> Tuple[Any, str]:
    encs: List[str] = []
    if preferred_encoding:
        encs.append(preferred_encoding)
    for e in _DEFAULT_ENCODINGS:
        if e not in encs:
            encs.append(e)

    last_err: Optional[Exception] = None
    for enc in encs:
        try:
            return path.open("r", encoding=enc, newline=""), enc
        except Exception as e:
            last_err = e
    raise last_err or OSError(f"Unable to open file: {path}")


def _parse_hms_seconds(x: Any) -> Optional[float]:
    """Parse HH:MM:SS(.mmm) to seconds; return None if not parseable."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or ":" not in s:
        return None
    parts = s.split(":")
    if len(parts) != 3:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        sec = float(parts[2])
        if not (0 <= m < 60):
            return None
        if not (0 <= sec < 61.0):
            return None
        return h * 3600.0 + m * 60.0 + sec
    except Exception:
        return None


def _parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
        return v if (math.isfinite(v) or math.isnan(v)) else None
    except Exception:
        return None


def _nondecreasing_ratio(seq: List[float]) -> float:
    vals = [v for v in seq if v is not None and not math.isnan(v)]
    if len(vals) < 3:
        return 0.0
    ok = 0
    for i in range(1, len(vals)):
        if vals[i] >= vals[i - 1]:
            ok += 1
    return ok / float(len(vals) - 1)


# -----------------------------
# Main detector
# -----------------------------
class ColumnRoleDetector:
    """
    Detect column roles for each CSV referenced by catalog_df.
    """

    def __init__(self, cfg: ExperimentConfig, *, output_dir: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir) if output_dir is not None else Path(cfg.target_root)

    def run(
        self,
        catalog_df: pd.DataFrame,
        spec: ColumnRoleSpec,
        *,
        write: bool = True,
        role_map_name: str = "column_roles.json",
        default_delimiter: str = ",",
    ) -> ColumnRoleArtifacts:
        required = {"mid", "raw_path", "raw_exists"}
        if not required.issubset(set(catalog_df.columns)):
            raise ValueError(f"catalog_df must contain columns: {sorted(required)}")

        sys_hints = {_norm_name(x) for x in spec.time_system_candidates}
        rel_hints = {_norm_name(x) for x in spec.time_relative_candidates}

        role_map: Dict[str, Any] = {}

        for _, row in catalog_df.iterrows():
            mid = str(row["mid"])
            raw_path = Path(str(row["raw_path"]))
            raw_exists = bool(row["raw_exists"])

            delimiter = str(row.get("csv_delimiter") or default_delimiter)
            enc_pref = row.get("csv_encoding", None)
            enc_pref = str(enc_pref).strip() if enc_pref is not None and str(enc_pref).strip() else None

            drop_extra = bool(row.get("csv_extra_field_candidate", False))
            extra_const = row.get("csv_extra_field_constant_value", None)
            dialect_ok = row.get("csv_dialect_ok", None)
            csv_read_ok = row.get("csv_read_ok", None)
            resolve_tag = row.get("resolve_tag", None)

            entry = {
                "mid": mid,
                "raw_path": str(raw_path),
                "raw_exists": raw_exists,
                "resolve_tag": resolve_tag,
                "status": "ok",
                "notes": [],
                "drop_extra_field": drop_extra,
                "extra_field_constant_value": extra_const,
                "read_params": {"delimiter": delimiter, "encoding_preferred": enc_pref},
                "roles": {"t_sys": None, "t_rel": None, "channels": []},
            }

            if dialect_ok is False and not drop_extra:
                entry["notes"].append("csv_dialect_ok=False (inspector). Proceeding best-effort.")

            if csv_read_ok is False:
                entry["status"] = "error"
                entry["notes"].append("csv_read_ok=False from dialect inspector; skipping role detection.")
                role_map[mid] = entry
                continue

            if not raw_exists:
                entry["status"] = "missing"
                entry["notes"].append("Raw file does not exist at raw_path.")
                role_map[mid] = entry
                continue

            try:
                header, samples = self._read_header_and_samples(
                    raw_path,
                    delimiter=delimiter,
                    encoding_preferred=enc_pref,
                    detect_rows=spec.detect_rows,
                    drop_extra_candidate=drop_extra,
                )
                if not header or not samples:
                    entry["status"] = "error"
                    entry["notes"].append("Empty header or insufficient valid rows for detection.")
                    role_map[mid] = entry
                    continue

                res = self._resolve_roles(
                    header=header,
                    sample_rows=samples,
                    sys_hints=sys_hints,
                    rel_hints=rel_hints,
                    spec=spec,
                )
                entry["roles"]["t_sys"] = res["t_sys"]
                entry["roles"]["t_rel"] = res["t_rel"]
                entry["roles"]["channels"] = res["channels"]
                entry["notes"].extend(res.get("notes", []))

                if spec.require_two_time_cols:
                    if res["t_sys"] is None or res["t_rel"] is None:
                        entry["status"] = "error"
                        entry["notes"].append("Failed to identify both time columns (t_sys and t_rel).")
                    elif int(res["t_sys"]["index"]) == int(res["t_rel"]["index"]):
                        entry["status"] = "error"
                        entry["notes"].append("Only one time-like column detected; t_sys and t_rel collide.")

            except Exception as e:
                entry["status"] = "error"
                entry["notes"].append(f"Exception: {type(e).__name__}: {e}")

            role_map[mid] = entry

        out_path = self.output_dir / role_map_name
        if write:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(role_map, indent=2, ensure_ascii=False), encoding="utf-8")

        return ColumnRoleArtifacts(role_map_path=out_path)

    def _read_header_and_samples(
        self,
        path: Path,
        *,
        delimiter: str,
        encoding_preferred: Optional[str],
        detect_rows: int,
        drop_extra_candidate: bool,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Streaming read: header + up to detect_rows valid rows.
        If drop_extra_candidate=True, drop trailing field only when row has header_ncols+1 fields.
        """
        f, _ = _open_text(path, preferred_encoding=encoding_preferred)
        with f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                return [], []

            header = [h.strip() for h in header]
            n = len(header)
            if n == 0:
                return [], []

            samples: List[List[str]] = []
            for row in reader:
                if not row:
                    continue
                if drop_extra_candidate and len(row) == n + 1:
                    row = row[:-1]
                if len(row) != n:
                    continue
                samples.append(row)
                if len(samples) >= detect_rows:
                    break

        return header, samples

    def _resolve_roles(
        self,
        *,
        header: List[str],
        sample_rows: List[List[str]],
        sys_hints: set,
        rel_hints: set,
        spec: ColumnRoleSpec,
    ) -> Dict[str, Any]:
        notes: List[str] = []
        n_cols = len(header)

        # Column-wise sample lists
        col_samples: List[List[str]] = [[] for _ in range(n_cols)]
        for r in sample_rows:
            if len(r) != n_cols:
                continue
            for j in range(n_cols):
                col_samples[j].append(r[j])

        # (1) Time-like candidates via HH:MM:SS parse ratio
        time_candidates: List[int] = []
        time_stats: Dict[int, Dict[str, Any]] = {}
        for j in range(n_cols):
            parsed = [_parse_hms_seconds(v) for v in col_samples[j]]
            ok = [p for p in parsed if p is not None]
            ratio = len(ok) / max(1, len(col_samples[j]))
            if ratio >= spec.min_parse_ratio_time and len(ok) >= 5:
                time_candidates.append(j)
                time_stats[j] = {
                    "parse_ratio": ratio,
                    "mono_ratio": _nondecreasing_ratio(ok),
                    "start_sec": ok[0],
                }

        def _hinted(hints: set) -> List[int]:
            out: List[int] = []
            for j, name in enumerate(header):
                if _norm_name(name) in hints and j in time_candidates:
                    out.append(j)
            return out

        sys_hint = _hinted(sys_hints)
        rel_hint = _hinted(rel_hints)

        t_sys_idx: Optional[int] = None
        t_rel_idx: Optional[int] = None

        # Priority: validated hints
        if sys_hint:
            t_sys_idx = sys_hint[0]
            notes.append("t_sys selected by hint + time-parse validation.")
        if rel_hint:
            t_rel_idx = rel_hint[0]
            notes.append("t_rel selected by hint + time-parse validation.")

        # Data heuristics
        if time_candidates:
            cand_sorted = sorted(
                time_candidates,
                key=lambda j: (time_stats[j]["mono_ratio"], time_stats[j]["parse_ratio"]),
                reverse=True,
            )

            # t_rel: start near 0 + monotonic preference
            if t_rel_idx is None:
                best_j, best_score = None, -1.0
                for j in cand_sorted:
                    start = time_stats[j]["start_sec"]
                    mono = time_stats[j]["mono_ratio"]
                    bonus = 1.0 if (start is not None and start <= 5.0) else 0.0
                    score = mono + bonus
                    if score > best_score:
                        best_j, best_score = j, score
                t_rel_idx = best_j

            # t_sys: best remaining distinct candidate
            if t_sys_idx is None:
                for j in cand_sorted:
                    if j != t_rel_idx:
                        t_sys_idx = j
                        break
                if t_sys_idx is None:
                    t_sys_idx = t_rel_idx  # only one time-like column exists

        # (2) Numeric channel candidates (exclude time columns)
        excluded = {t_sys_idx, t_rel_idx}
        numeric_idxs: List[int] = []
        for j in range(n_cols):
            if j in excluded:
                continue
            parsed = [_parse_float(v) for v in col_samples[j]]
            ok = [p for p in parsed if p is not None]
            ratio = len(ok) / max(1, len(col_samples[j]))
            if ratio >= spec.min_parse_ratio_numeric and len(ok) >= 5:
                numeric_idxs.append(j)
        numeric_idxs.sort()

        pool = list(spec.channel_name_pool)
        channels: List[Dict[str, Any]] = []
        for k, j in enumerate(numeric_idxs):
            cname = pool[k] if k < len(pool) else f"Ch{k+1}"
            channels.append({"index": int(j), "raw_name": header[j], "canonical_name": cname})
        if not channels:
            notes.append("No numeric channel columns detected (thresholds may be too strict).")

        t_sys = None if t_sys_idx is None else {"index": int(t_sys_idx), "raw_name": header[t_sys_idx], "canonical_name": "t_sys"}
        t_rel = None if t_rel_idx is None else {"index": int(t_rel_idx), "raw_name": header[t_rel_idx], "canonical_name": "t_rel"}

        return {"t_sys": t_sys, "t_rel": t_rel, "channels": channels, "notes": notes}
