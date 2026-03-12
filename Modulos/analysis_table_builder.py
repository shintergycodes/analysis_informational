# analysis_table_builder.py
"""
Analysis Table Builder (refactored)

Primary responsibility
----------------------
Build "Analysis Ready" Parquet tables (one per measurement) from:
- an analysis catalog DataFrame (typically dialect-enriched by upstream stages)
- column_roles.json (produced by column_role_detection.py)

Design constraints
------------------
- Read-only with respect to raw CSVs (never edits raw files).
- Experimental-mode agnostic (double-blind vs declared does not affect this module).
- Low-resource friendly: processes one file at a time; streaming IO is delegated to
  analysis_table_io.py (to be added next).
Outputs
-------
- Parquet files under (v3 default):
    <root>/Analysis Ready/<fecha>/<lab>/<filename>.parquet
- Legacy-compatible optional layout:
    <root>/Analysis Ready/<fecha>/<lab>/<jornada>/<filename>.parquet
- JSONL actions log:
    <root>/table_actions.jsonl
Notes
-----
This file intentionally delegates CSV selection, cleaning, and (optional) streaming
Parquet writing to analysis_table_io.py to keep this module small and maintainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import json
import time

import pandas as pd

OutputLayout = Literal["fecha_lab", "fecha_lab_jornada", "flat"]
FilenameStrategy = Literal["mid", "archivo_color"]


# -----------------------------
# Optional dependency (added next)
# -----------------------------
try:
    # Will be created in the next step of the refactor.
    from analysis_table_io import build_dataframe_from_csv, write_parquet_from_csv  # type: ignore
except Exception:  # pragma: no cover
    build_dataframe_from_csv = None  # type: ignore
    write_parquet_from_csv = None  # type: ignore


@dataclass(frozen=True)
class TableBuildPolicy:
    """
    Deterministic build policy.

    sentinels:
        Numeric sentinel values treated as missing in channel columns (-> NaN).
    default_delimiter:
        Used if catalog does not provide csv_delimiter.
    encoding_candidates:
        Used if catalog does not provide csv_encoding.
    output_layout:
        Folder layout contract for outputs.
    filename_strategy:
        How to name parquet files:
          - "mid": <mid>.parquet (preferred when mid is Windows-safe)
          - "archivo_color": <archivo:02d>med<color>.parquet (legacy)
    overwrite:
        If False, existing parquet files are not overwritten.
    """
    sentinels: Tuple[float, ...] = (-111.0,)
    default_delimiter: str = ","
    encoding_candidates: Tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")
    output_layout: OutputLayout = "fecha_lab"
    filename_strategy: FilenameStrategy = "mid"
    overwrite: bool = True


@dataclass(frozen=True)
class TableBuildArtifacts:
    output_root: Path
    actions_path: Path


class AnalysisTableBuilder:
    """
    Build Analysis Ready tables from catalog + roles.

    Expected catalog columns (minimum):
      - mid, fecha, lab, jornada, raw_path, raw_exists
    Optional (dialect-enriched):
      - csv_extra_field_candidate, csv_delimiter, csv_encoding
    Optional (legacy filename):
      - archivo, color   (required only if filename_strategy="archivo_color")
    """

    BASE_REQUIRED_COLS = ("mid", "fecha", "lab", "jornada", "raw_path", "raw_exists")

    def __init__(
        self,
        root: Path,
        *,
        output_dir_name: str = "Analysis Ready",
        actions_name: str = "table_actions.jsonl",
    ) -> None:
        self.root = Path(root)
        self.output_root = self.root / output_dir_name
        self.actions_path = self.root / actions_name

    # -------------------------
    # Public API
    # -------------------------
    def run(
        self,
        catalog_df: pd.DataFrame,
        *,
        column_roles_path: Path,
        policy: Optional[TableBuildPolicy] = None,
        write_actions: bool = True,
    ) -> TableBuildArtifacts:
        pol = policy or TableBuildPolicy()
        self._validate_catalog(catalog_df, pol)

        roles_map = self._load_roles_map(Path(column_roles_path))

        self.output_root.mkdir(parents=True, exist_ok=True)
        if write_actions:
            self.actions_path.parent.mkdir(parents=True, exist_ok=True)

        # itertuples is faster and lighter than iterrows for large catalogs
        if write_actions and pol.overwrite and self.actions_path.exists():
            self.actions_path.unlink(missing_ok=True)
        
        for row in catalog_df.itertuples(index=False):
           
           
            action = self._process_one(row=row, roles_map=roles_map, policy=pol)
            if write_actions:
                with self.actions_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(action, ensure_ascii=False) + "\n")

        return TableBuildArtifacts(output_root=self.output_root, actions_path=self.actions_path)

    # -------------------------
    # Validation / loading
    # -------------------------
    def _validate_catalog(self, df: pd.DataFrame, policy: TableBuildPolicy) -> None:
        missing = [c for c in self.BASE_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Catalog DataFrame missing required columns: {missing}")

        if policy.filename_strategy == "archivo_color":
            extra_missing = [c for c in ("archivo", "color") if c not in df.columns]
            if extra_missing:
                raise ValueError(
                    "filename_strategy='archivo_color' requires catalog columns: "
                    f"{extra_missing}"
                )

    @staticmethod
    def _load_roles_map(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"column_roles.json not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("column_roles.json must be a JSON object (dict).")
        return data

    # -------------------------
    # Per-row processing
    # -------------------------
    def _process_one(self, *, row: Any, roles_map: Dict[str, Any], policy: TableBuildPolicy) -> Dict[str, Any]:
        t0 = time.time()

        mid = str(getattr(row, "mid"))
        fecha = str(getattr(row, "fecha"))
        lab = str(getattr(row, "lab"))
        jornada = str(getattr(row, "jornada"))
        raw_path = Path(str(getattr(row, "raw_path")))
        raw_exists = bool(getattr(row, "raw_exists"))

        # Dialect-enriched optional fields
        drop_extra_catalog = getattr(row, "csv_extra_field_candidate", None)
        delimiter = getattr(row, "csv_delimiter", None) or policy.default_delimiter
        encoding = getattr(row, "csv_encoding", None)
        encoding = str(encoding).strip() if encoding is not None and str(encoding).strip() else None

        action: Dict[str, Any] = {
            "mid": mid,
            "fecha": fecha,
            "lab": lab,
            "jornada": jornada,
            "raw_path": str(raw_path),
            "output_path": None,
            "status": "init",
            "applied": {
                "drop_extra_field": None,
                "filename_strategy": policy.filename_strategy,
                "selected_indices": [],
                "renamed_columns": {},
                "sentinels_to_nan": list(policy.sentinels),
            },
            "counts": {},
            "notes": [],
            "errors": [],
            "timing_sec": None,
        }

        if not raw_exists or not raw_path.exists():
            action["status"] = "skipped_missing_raw"
            action["errors"].append("missing_raw_file")
            action["timing_sec"] = round(time.time() - t0, 6)
            return action

        role_entry = roles_map.get(mid)
        if role_entry is None:
            action["status"] = "error"
            action["errors"].append("missing_roles_for_mid")
            action["timing_sec"] = round(time.time() - t0, 6)
            return action

        if str(role_entry.get("status", "ok")) != "ok":
            action["status"] = "error"
            action["errors"].append(f"roles_status_not_ok:{role_entry.get('status')}")
            action["timing_sec"] = round(time.time() - t0, 6)
            return action

        try:
            # Prefer catalog flag; fall back to roles flag if catalog doesn't have it.
            drop_extra_roles = bool(role_entry.get("drop_extra_field", False))
            drop_extra = bool(drop_extra_catalog) if drop_extra_catalog is not None else drop_extra_roles
            action["applied"]["drop_extra_field"] = drop_extra

            spec = self._extract_role_spec(role_entry)
            indices: List[int] = spec["indices"]
            out_columns: List[str] = spec["out_columns"]
            channel_columns: List[str] = spec["channel_columns"]
            renamed_columns: Dict[str, Optional[str]] = spec["renamed_columns"]

            action["applied"]["selected_indices"] = indices
            action["applied"]["renamed_columns"] = renamed_columns

            out_path = self._build_output_path(
                fecha=fecha,
                lab=lab,
                jornada=jornada,
                mid=mid,
                row=row,
                policy=policy,
            )
            action["output_path"] = str(out_path)

            if out_path.exists() and not policy.overwrite:
                action["status"] = "skipped_exists"
                action["notes"].append("output_exists_and_overwrite_false")
                action["timing_sec"] = round(time.time() - t0, 6)
                return action

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Delegate IO/cleaning to analysis_table_io.py
            if write_parquet_from_csv is not None:
                stats = write_parquet_from_csv(
                    csv_path=raw_path,
                    parquet_path=out_path,
                    indices=indices,
                    out_columns=out_columns,
                    channel_columns=channel_columns,
                    sentinels=set(policy.sentinels),
                    delimiter=str(delimiter),
                    encoding=encoding,
                    encoding_candidates=policy.encoding_candidates,
                    drop_extra_field=drop_extra,
                )
                if isinstance(stats, dict):
                    action["counts"].update(stats)
            elif build_dataframe_from_csv is not None:
                df, stats = build_dataframe_from_csv(
                    csv_path=raw_path,
                    indices=indices,
                    out_columns=out_columns,
                    channel_columns=channel_columns,
                    sentinels=set(policy.sentinels),
                    delimiter=str(delimiter),
                    encoding=encoding,
                    encoding_candidates=policy.encoding_candidates,
                    drop_extra_field=drop_extra,
                )
                df.to_parquet(out_path, index=False)
                if isinstance(stats, dict):
                    action["counts"].update(stats)
            else:
                raise ImportError(
                    "analysis_table_io.py is missing. Create it next with "
                    "build_dataframe_from_csv(...) and/or write_parquet_from_csv(...)."
                )

            action["status"] = "ok"
            action["timing_sec"] = round(time.time() - t0, 6)
            return action

        except Exception as e:
            action["status"] = "error"
            action["errors"].append(f"{type(e).__name__}: {e}")
            action["timing_sec"] = round(time.time() - t0, 6)
            return action

    # -------------------------
    # Role contract extraction
    # -------------------------
    def _extract_role_spec(self, role_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        role_entry is expected to come from column_roles.json:
          - role_entry["roles"]["t_sys"] / ["t_rel"] are dicts with keys: index, raw_name
          - role_entry["roles"]["channels"] is a list of dicts with keys: index, canonical_name, raw_name
        """
        roles = role_entry.get("roles", {})
        if not isinstance(roles, dict):
            raise ValueError("Invalid roles schema: roles must be a dict")

        def idx_of(obj: Any) -> Optional[int]:
            if not isinstance(obj, dict):
                return None
            try:
                i = int(obj.get("index"))
                return i if i >= 0 else None
            except Exception:
                return None

        t_sys = roles.get("t_sys")
        t_rel = roles.get("t_rel")
        channels = roles.get("channels", [])

        t_sys_idx = idx_of(t_sys)
        t_rel_idx = idx_of(t_rel)

        if t_sys_idx is None and t_rel_idx is None:
            raise ValueError("No valid time indices found (t_sys and t_rel are missing).")

        if not isinstance(channels, list):
            raise ValueError("Invalid roles schema: channels must be a list")

        out_columns: List[str] = []
        indices: List[int] = []
        renamed_columns: Dict[str, Optional[str]] = {}
        channel_columns: List[str] = []

        if t_sys_idx is not None:
            out_columns.append("t_sys")
            indices.append(t_sys_idx)
            renamed_columns["t_sys"] = t_sys.get("raw_name") if isinstance(t_sys, dict) else None

        if t_rel_idx is not None:
            out_columns.append("t_rel")
            indices.append(t_rel_idx)
            renamed_columns["t_rel"] = t_rel.get("raw_name") if isinstance(t_rel, dict) else None

        for ch in channels:
            if not isinstance(ch, dict):
                continue
            cname = str(ch.get("canonical_name", "")).strip()
            if not cname:
                continue
            cidx = idx_of(ch)
            if cidx is None:
                continue
            out_columns.append(cname)
            indices.append(cidx)
            renamed_columns[cname] = ch.get("raw_name")
            channel_columns.append(cname)

        if not channel_columns:
            raise ValueError("No channel columns found in roles.")

        return {
            "out_columns": out_columns,
            "indices": indices,
            "renamed_columns": renamed_columns,
            "channel_columns": channel_columns,
        }

    # -------------------------
    # Output path contract
    # -------------------------
    def _build_output_path(
        self,
        *,
        fecha: str,
        lab: str,
        jornada: str,
        mid: str,
        row: Any,
        policy: TableBuildPolicy,
    ) -> Path:
        filename = self._build_filename(mid=mid, row=row, policy=policy)

        if policy.output_layout == "fecha_lab":
            return self.output_root / fecha / lab / filename

        if policy.output_layout == "fecha_lab_jornada":
            return self.output_root / fecha / lab / jornada / filename

        return self.output_root / filename

    @staticmethod
    def _build_filename(*, mid: str, row: Any, policy: TableBuildPolicy) -> str:
        if policy.filename_strategy == "mid":
            return f"{mid}.parquet"

        # Legacy naming strategy: <archivo:02d>med<color>.parquet
        try:
            archivo = int(getattr(row, "archivo"))
        except Exception:
            archivo = 0
        color = str(getattr(row, "color", "UNK"))
        return f"{archivo:02d}med{color}.parquet"
