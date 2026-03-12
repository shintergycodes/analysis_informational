# measurement_time_bounds.py
"""
Measurement Time Bounds (part 2/2)

Goal
----
Using the previously detected column roles (column_roles.json), extract per CSV:
- t_sys_start, t_sys_end
- t_rel_start, t_rel_end

This module:
- DOES NOT re-detect roles.
- DOES NOT rewrite CSVs.
- Streams reads (low-RAM friendly).
- Respects the "drop_extra_field" flag from column_roles.json by conditionally dropping
  a trailing field only when a row has exactly (header_ncols + 1) fields.

Inputs
------
1) catalog_df (pandas.DataFrame), typically output of CSVDialectInspector.run(...)
   Required columns: mid, raw_path, raw_exists
   Optional columns: csv_delimiter, csv_encoding, csv_dialect_ok

2) column_roles.json (output of column_role_detection.py)
   Expected keys per mid:
     - status
     - raw_exists
     - drop_extra_field
     - roles.t_sys.index
     - roles.t_rel.index

Outputs
-------
- measurement_time_bounds.json (written to cfg.target_root by default)

JSON Format
-----------
A dict keyed by mid:
{
  "<mid>": {
    "mid": "...",
    "raw_path": "...",
    "raw_exists": true,
    "status": "ok|missing|error",
    "notes": [...],
    "t_sys_start": "...|null",
    "t_sys_end":   "...|null",
    "t_rel_start": "...|null",
    "t_rel_end":   "...|null"
  },
  ...
}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import json

import pandas as pd

from config import ExperimentConfig


# -----------------------------
# Artifacts
# -----------------------------
@dataclass(frozen=True)
class TimeBoundsArtifacts:
    time_bounds_path: Path


# -----------------------------
# Helpers
# -----------------------------
_DEFAULT_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")


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


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, int):
            return int(x)
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def _get_role_index(role_obj: Any) -> int:
    """
    role_obj is expected like:
      {"index": 0, "raw_name": "...", "canonical_name": "t_sys"}
    """
    if not isinstance(role_obj, dict):
        return -1
    return _safe_int(role_obj.get("index"), default=-1)


# -----------------------------
# Core builder
# -----------------------------
class MeasurementTimeBounds:
    """
    Extract time bounds using column_roles.json and catalog_df.
    """

    def __init__(self, cfg: ExperimentConfig, *, output_dir: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir) if output_dir is not None else Path(cfg.target_root)

    def run(
        self,
        catalog_df: pd.DataFrame,
        *,
        roles_path: Optional[Path] = None,
        write: bool = True,
        output_name: str = "measurement_time_bounds.json",
        default_delimiter: str = ",",
    ) -> TimeBoundsArtifacts:
        required = {"mid", "raw_path", "raw_exists"}
        if not required.issubset(set(catalog_df.columns)):
            raise ValueError(f"catalog_df must contain columns: {sorted(required)}")

        rp = Path(roles_path) if roles_path is not None else (Path(self.cfg.target_root) / "column_roles.json")
        if not rp.exists():
            raise FileNotFoundError(f"column_roles.json not found: {rp}")

        roles_map = json.loads(rp.read_text(encoding="utf-8"))

        bounds_map: Dict[str, Any] = {}

        for _, row in catalog_df.iterrows():
            mid = str(row["mid"])
            raw_path = Path(str(row["raw_path"]))
            raw_exists = bool(row["raw_exists"])

            delimiter = str(row.get("csv_delimiter") or default_delimiter)
            enc_pref = row.get("csv_encoding", None)
            enc_pref = str(enc_pref).strip() if enc_pref is not None and str(enc_pref).strip() else None

            role_entry = roles_map.get(mid, None)

            entry = {
                "mid": mid,
                "raw_path": str(raw_path),
                "raw_exists": raw_exists,
                "status": "ok",
                "notes": [],
                "t_sys_start": None,
                "t_sys_end": None,
                "t_rel_start": None,
                "t_rel_end": None,
            }

            if role_entry is None:
                entry["status"] = "error"
                entry["notes"].append("Missing entry in column_roles.json for this mid.")
                bounds_map[mid] = entry
                continue

            role_status = str(role_entry.get("status", "ok"))
            drop_extra = bool(role_entry.get("drop_extra_field", False))
            roles_obj = role_entry.get("roles", {}) if isinstance(role_entry.get("roles", {}), dict) else {}

            t_sys_idx = _get_role_index(roles_obj.get("t_sys"))
            t_rel_idx = _get_role_index(roles_obj.get("t_rel"))

            if role_status != "ok":
                entry["status"] = role_status
                entry["notes"].append(f"Roles status is '{role_status}'. Bounds will not be extracted.")
                bounds_map[mid] = entry
                continue

            if not raw_exists:
                entry["status"] = "missing"
                entry["notes"].append("Raw file does not exist at raw_path.")
                bounds_map[mid] = entry
                continue

            if t_sys_idx < 0 and t_rel_idx < 0:
                entry["status"] = "error"
                entry["notes"].append("No valid time indices found in roles (t_sys and t_rel are missing).")
                bounds_map[mid] = entry
                continue

            # Stream read: header + first valid row + last valid row
            try:
                header, first_row, last_row = self._read_first_last_valid_rows(
                    raw_path,
                    delimiter=delimiter,
                    encoding_preferred=enc_pref,
                    drop_extra_candidate=drop_extra,
                )

                if not header:
                    entry["status"] = "error"
                    entry["notes"].append("Empty header or unreadable CSV.")
                    bounds_map[mid] = entry
                    continue

                n = len(header)
                if first_row is None or last_row is None:
                    entry["status"] = "error"
                    entry["notes"].append("No valid data rows found (after applying row-width rules).")
                    bounds_map[mid] = entry
                    continue

                # Extract bounds as raw strings (no parsing here)
                if 0 <= t_sys_idx < n:
                    entry["t_sys_start"] = first_row[t_sys_idx]
                    entry["t_sys_end"] = last_row[t_sys_idx]
                elif t_sys_idx >= 0:
                    entry["notes"].append(f"t_sys index out of range: {t_sys_idx} (ncols={n})")

                if 0 <= t_rel_idx < n:
                    entry["t_rel_start"] = first_row[t_rel_idx]
                    entry["t_rel_end"] = last_row[t_rel_idx]
                elif t_rel_idx >= 0:
                    entry["notes"].append(f"t_rel index out of range: {t_rel_idx} (ncols={n})")

                # Minimal completeness notes
                if entry["t_sys_start"] is None or entry["t_sys_end"] is None:
                    entry["notes"].append("t_sys bounds incomplete.")
                if entry["t_rel_start"] is None or entry["t_rel_end"] is None:
                    entry["notes"].append("t_rel bounds incomplete.")

            except Exception as e:
                entry["status"] = "error"
                entry["notes"].append(f"Exception: {type(e).__name__}: {e}")

            bounds_map[mid] = entry

        out_path = self.output_dir / output_name
        if write:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(bounds_map, indent=2, ensure_ascii=False), encoding="utf-8")

        return TimeBoundsArtifacts(time_bounds_path=out_path)

    def _read_first_last_valid_rows(
        self,
        path: Path,
        *,
        delimiter: str,
        encoding_preferred: Optional[str],
        drop_extra_candidate: bool,
    ) -> Tuple[List[str], Optional[List[str]], Optional[List[str]]]:
        """
        Stream read the file and return:
          - header
          - first valid data row
          - last valid data row

        Valid row definition:
          - if drop_extra_candidate and len(row) == header_ncols + 1: drop last field
          - row length must match header_ncols after the conditional drop
        """
        f, _enc = _open_text(path, preferred_encoding=encoding_preferred)
        with f:
            reader = csv.reader(f, delimiter=delimiter)

            try:
                header = next(reader)
            except StopIteration:
                return [], None, None

            header = [h.strip() for h in header]
            n = len(header)
            if n == 0:
                return [], None, None

            first_row: Optional[List[str]] = None
            last_row: Optional[List[str]] = None

            for row in reader:
                if not row:
                    continue

                if drop_extra_candidate and len(row) == n + 1:
                    row = row[:-1]

                if len(row) != n:
                    continue

                if first_row is None:
                    first_row = row
                last_row = row

        return header, first_row, last_row
