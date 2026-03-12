# analysis_ready_prep.py
"""
Tabular-Ready preparation stage.

This module creates two lightweight, reproducible artifacts in `cfg.target_root`:

1) `analysis_catalog.csv`
   A stable index built from `manifest_all.csv` that resolves each measurement to the
   *physical* CSV path that downstream modules should read.

2) `dialect_report.json` (optional, via `CSVDialectInspector`)
   A shallow structural inspection of each CSV (header vs rows column counts) to detect
   common issues (e.g., a constant "extra field" at the end of rows).

Design goals
------------
- Do NOT modify source data (`cfg.source_root`) nor `manifest_all.csv`.
- Be compatible with both experimental modes:
  - double-blind (label_mode="blind"): normalized filenames are typically MID-based.
  - declared/unblinded modes: filenames can follow legacy "{archivo}med{color}.csv" or renames.
- Keep memory usage low: stream CSV reads, avoid loading raw datasets into RAM.

Primary dependency: pandas (for catalog I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import csv
import json
from typing import Dict, List, Optional, Tuple, Literal, Any

import pandas as pd

from config import ExperimentConfig


ResolveMode = Literal["strict", "compat"]


# ---------------------------------------------------------------------
# (1) Analysis-ready catalog from manifest
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CatalogPaths:
    """Paths for artifacts produced by the catalog stage."""
    manifest_path: Path
    catalog_path: Path


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (int,)):
            return int(x)
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))  # tolerate "1.0"
    except Exception:
        return default


def _lower_if_ascii(s: str) -> str:
    # Lower only when this is a typical ASCII token; preserve case otherwise.
    # This helps keep filenames like "1medA.csv" stable while still normalizing "Negra" -> "negra".
    try:
        return s.lower() if s.isascii() else s
    except Exception:
        return s


class AnalysisCatalogBuilder:
    """
    Build `analysis_catalog.csv` from `manifest_all.csv`.

    Input
    -----
    - `manifest_all.csv` produced earlier in the pipeline (typically by `ManifestWriter`).

    Output
    ------
    - `analysis_catalog.csv` with stable columns used by downstream modules.

    Notes on blind vs declared
    --------------------------
    - In double-blind mode (cfg.label_mode == "blind"), the normalization step usually writes
      files as "{mid}.csv" where:
          mid = "{fecha}_{turno}_{lab}_{archivo:03d}"
      This builder follows that convention for `filename_expected`.
    - In declared/unblinded modes, the common legacy convention is "{archivo}med{color}.csv".
      This builder uses it when it can, but will also attempt lightweight directory heuristics
      in `resolve_mode="compat"`.

    Resolution strategy
    -------------------
    - strict: only the expected path is used (raw_exists may be False).
    - compat: if expected path is missing, try a few safe heuristics in the same directory.
    """

    REQUIRED_MANIFEST_COLS = ["fecha", "lab", "archivo"]

    def __init__(
        self,
        cfg: ExperimentConfig,
        manifest_path: Optional[Path] = None,
        catalog_path: Optional[Path] = None,
        *,
        raw_subdir_name: str = "Raw Data",
        resolve_mode_default: ResolveMode = "compat",
        prefer_mid_filenames_in_blind: bool = True,
        mid_format: str = "{fecha}_{lab}_{archivo:03d}",
    ) -> None:
        self.cfg = cfg
        self.raw_subdir_name = raw_subdir_name
        self.resolve_mode_default = resolve_mode_default
        self.prefer_mid_filenames_in_blind = prefer_mid_filenames_in_blind
        self.mid_format = mid_format

        mp = manifest_path if manifest_path is not None else (cfg.target_root / "manifest_all.csv")
        cp = catalog_path if catalog_path is not None else (cfg.target_root / "analysis_catalog.csv")
        self.paths = CatalogPaths(manifest_path=Path(mp), catalog_path=Path(cp))

        # Cache external mappings once (important on low-resource machines).
        self._labels_by_mid: Dict[str, str] = {}
        self._labels_by_name: Dict[str, str] = {}
        self._groups_by_date: Dict[str, str] = {}
        self._groups_by_mid: Dict[str, str] = {}
        self._groups_by_name: Dict[str, str] = {}
        self._load_external_mappings()

    # ----------------------------
    # External mapping loaders
    # ----------------------------
    def _load_external_mappings(self) -> None:
        # Labels
        if getattr(self.cfg, "label_mode", "") == "external" and getattr(self.cfg, "labels_csv", None):
            p = Path(self.cfg.labels_csv)
            if p.exists():
                self._labels_by_mid, self._labels_by_name = self._read_labels_csv(p)

        # Groups
        if getattr(self.cfg, "group_mode", "") == "external_ctrl_exp" and getattr(self.cfg, "groups_csv", None):
            p = Path(self.cfg.groups_csv)
            if p.exists():
                self._groups_by_date, self._groups_by_mid, self._groups_by_name = self._read_groups_csv(p)

    @staticmethod
    def _read_labels_csv(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
        by_mid: Dict[str, str] = {}
        by_name: Dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                label = _safe_str(r.get("label"))
                if not label:
                    continue
                mid = _safe_str(r.get("mid"))
                name = _safe_str(r.get("canonical_name") or r.get("filename") or r.get("file"))
                if mid:
                    by_mid[mid] = label
                if name:
                    by_name[name] = label
        return by_mid, by_name

    @staticmethod
    def _read_groups_csv(path: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        by_date: Dict[str, str] = {}
        by_mid: Dict[str, str] = {}
        by_name: Dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                group = _safe_str(r.get("group"))
                if not group:
                    continue
                date = _safe_str(r.get("date") or r.get("fecha"))
                mid = _safe_str(r.get("mid"))
                name = _safe_str(r.get("canonical_name") or r.get("filename") or r.get("file"))
                if date:
                    by_date[date] = group
                if mid:
                    by_mid[mid] = group
                if name:
                    by_name[name] = group
        return by_date, by_mid, by_name

    # ----------------------------
    # Helpers: mid / label / group
    # ----------------------------
    def build_mid(self, *, fecha: str, lab: str, archivo: int) -> str:
        return self.mid_format.format(
            fecha=fecha,
            lab=lab,
            archivo=archivo,
        )

    def label_for(self, *, mid: str, canonical_name: str) -> str:
        mode = getattr(self.cfg, "label_mode", "blind")
        default = getattr(self.cfg, "default_label", "UNK")

        if mode == "blind":
            return default

        if mode == "declared":
            m = getattr(self.cfg, "declared_label_map", {}) or {}
            return _safe_str(m.get(canonical_name)) or default

        if mode == "external":
            return self._labels_by_mid.get(mid) or self._labels_by_name.get(canonical_name) or default

        return default

    def group_for(self, *, fecha: str, mid: str, canonical_name: str) -> str:
        mode = getattr(self.cfg, "group_mode", "by_date")
        default = getattr(self.cfg, "default_group", "UNK")

        if mode == "by_date":
            m = getattr(self.cfg, "date_to_group", {}) or {}
            return _safe_str(m.get(fecha)) or default

        if mode == "declared_ctrl_exp":
            m = getattr(self.cfg, "date_to_group", {}) or {}
            return _safe_str(m.get(fecha)) or default

        if mode == "external_ctrl_exp":
            return (
                self._groups_by_date.get(fecha)
                or self._groups_by_mid.get(mid)
                or self._groups_by_name.get(canonical_name)
                or default
            )

        return default

    # ----------------------------
    # Expected filename and path resolution
    # ----------------------------
    def expected_filename(self, *, mid: str, archivo: int, color: str) -> str:
        """
        Expected output filename for the normalized raw CSV.

        - blind mode: prefer "{mid}.csv" (matches Normalizer default in v2).
        - declared/unblinded: "{archivo}med{color}.csv" when possible.
        """
        label_mode = getattr(self.cfg, "label_mode", "blind")

        if label_mode == "blind" and self.prefer_mid_filenames_in_blind:
            return f"{mid}.csv"

        # declared/unblinded path convention (legacy)
        if archivo >= 0 and color and color.upper() != "UNK":
            # Preserve original token casing for maximum compatibility.
            return f"{archivo}med{color}.csv"

        # Fallback: still return MID-based name (better than "UNK" filenames).
        return f"{mid}.csv"

    def _raw_dir(self, fecha: str, lab: str) -> Path:
        return self.cfg.source_root / fecha / lab / self.raw_subdir_name
    
    def _resolve_path(
        self,
        *,
        fecha: str,
        lab: str,
        mid: str,
        archivo: int,
        color: str,
        resolve_mode: ResolveMode,
    ) -> Tuple[Path, str]:
        """
        Resolve a CSV path under:/<lab>/<Raw Dat
            source_root/<fecha>/<lab>/<Raw Data>    

        Returns:
            (resolved_path, resolve_tag)

        Tags:
            expected, compat_found, missing, missing_dir
        """
        raw_dir = self._raw_dir(fecha, lab)
        expected_name = self.expected_filename(mid=mid, archivo=archivo, color=color)
        expected_path = raw_dir / expected_name

        if expected_path.exists() or resolve_mode == "strict":
            return expected_path, "expected" if expected_path.exists() else "missing"

        if not raw_dir.exists():
            # Keep expected path for traceability.
            return expected_path, "missing_dir"

        # compat heuristics (small directories: usually <= 9 files)
        candidates: List[Path] = []

        # 1) If blind: try MID-based even if expected wasn't (defensive).
        mid_path = raw_dir / f"{mid}.csv"
        if mid_path.exists():
            return mid_path, "compat_found"

        # 2) Legacy padding variants (some older exports used 02d/03d).
        if archivo >= 0 and color and color.upper() != "UNK":
            candidates.extend(
                [
                    raw_dir / f"{archivo:02d}med{color}.csv",
                    raw_dir / f"{archivo:03d}med{color}.csv",
                    raw_dir / f"{archivo:02d}med{_lower_if_ascii(color)}.csv",
                    raw_dir / f"{archivo:03d}med{_lower_if_ascii(color)}.csv",
                ]
            )
            for p in candidates:
                if p.exists():
                    return p, "compat_found"

        # 3) Directory scan heuristics (case-insensitive match)
        try:
            csv_files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
        except Exception:
            csv_files = []

        if not csv_files:
            return expected_path, "missing"

        # 3a) match by "{archivo}med" prefix
        if archivo >= 0:
            pref = f"{archivo}med"
            pref_lower = pref.lower()
            pref_matches = [p for p in csv_files if p.name.lower().startswith(pref_lower)]
            if len(pref_matches) == 1:
                return pref_matches[0], "compat_found"

            # 3b) match by numeric prefix only
            num_matches = []
            for p in csv_files:
                stem = p.stem
                # extract leading digits
                i = 0
                while i < len(stem) and stem[i].isdigit():
                    i += 1
                if i > 0:
                    try:
                        n = int(stem[:i])
                    except Exception:
                        n = None
                    if n == archivo:
                        num_matches.append(p)
            if len(num_matches) == 1:
                return num_matches[0], "compat_found"

        # Not found: return expected for traceability.
        return expected_path, "missing"

    # ----------------------------
    # Build catalog
    # ----------------------------
    def build(self, resolve_mode: Optional[ResolveMode] = None, write: bool = True) -> pd.DataFrame:
        """
        Build the analysis catalog and (optionally) write it to disk.

        Output columns (stable):
          mid, fecha, jornada, lab, turno, archivo, color, etiqueta,
          filename_expected, canonical_name,
          raw_path, raw_exists, resolve_tag, relpath,
          jornada_manifest, etiqueta_manifest
        """
        rm = resolve_mode or self.resolve_mode_default

        mp = self.paths.manifest_path
        if not mp.exists():
            raise FileNotFoundError(f"manifest_all.csv not found: {mp}")

        manifest = pd.read_csv(mp)

        missing_cols = [c for c in self.REQUIRED_MANIFEST_COLS if c not in manifest.columns]
        if missing_cols:
            raise ValueError(f"Manifest is missing required columns: {missing_cols}")

        # Minimal type normalization (do NOT enforce lowercase filenames).
        mf = manifest.copy()
        mf["fecha"] = mf["fecha"].astype(str).map(_safe_str)
        mf["lab"] = mf["lab"].astype(str).map(_safe_str)
        mf["archivo"] = pd.to_numeric(mf["archivo"], errors="coerce").fillna(-1).astype(int)

        if "turno" not in mf.columns:
            mf["turno"] = ""
        else:
            mf["turno"] = mf["turno"].astype(str).map(_safe_str)

        if "jornada" not in mf.columns:
            mf["jornada"] = "UNK"
        else:
            mf["jornada"] = mf["jornada"].astype(str).map(_safe_str)

        if "etiqueta" not in mf.columns:
            mf["etiqueta"] = "UNK"
        else:
            mf["etiqueta"] = mf["etiqueta"].astype(str).map(_safe_str)

        if "color" not in mf.columns:
            mf["color"] = "UNK"
        else:
            mf["color"] = mf["color"].astype(str).map(_safe_str)
            
        rows: List[Dict[str, object]] = []

        for _, r in mf.iterrows():
            fecha = str(r["fecha"])
            lab = str(r["lab"])
            turno = str(r["turno"])
            archivo = int(r["archivo"])
            color = str(r["color"]) if r["color"] else "UNK"

            mid = self.build_mid(fecha=fecha, lab=lab, archivo=archivo)

            resolved_path, tag = self._resolve_path(
                fecha=fecha,
                lab=lab,
                mid=mid,
                archivo=archivo,
                color=color,
                resolve_mode=rm,
            )
            exists = resolved_path.exists()

            # The canonical name is what downstream labeling maps typically use.
            # Prefer the resolved filename if it exists; else fall back to expected.
            filename_expected = self.expected_filename(mid=mid, archivo=archivo, color=color)
            canonical_name = resolved_path.name if exists else filename_expected

            jornada_cfg = self.group_for(fecha=fecha, mid=mid, canonical_name=canonical_name)
            etiqueta_cfg = self.label_for(mid=mid, canonical_name=canonical_name)

            relpath = ""
            try:
                relpath = resolved_path.relative_to(self.cfg.target_root).as_posix()
            except Exception:
                relpath = resolved_path.as_posix()

            rows.append(
                {
                    "mid": mid,
                    "fecha": fecha,
                    # `jornada` and `etiqueta` reflect the *current* config (supports late unblinding)
                    "jornada": jornada_cfg,
                    "lab": lab,
                    "turno": turno,
                    "archivo": archivo,
                    "color": color,
                    "etiqueta": etiqueta_cfg,
                    "filename_expected": filename_expected,
                    "canonical_name": canonical_name,
                    "raw_path": str(resolved_path),
                    "raw_exists": bool(exists),
                    "resolve_tag": tag,
                    "relpath": relpath,
                    # preserve what was recorded at manifest time (useful for audits)
                    "jornada_manifest": str(r["jornada"]),
                    "etiqueta_manifest": str(r["etiqueta"]),
                }
            )

        catalog_df = pd.DataFrame(rows)

        if write:
            out = self.paths.catalog_path
            out.parent.mkdir(parents=True, exist_ok=True)
            catalog_df.to_csv(out, index=False, encoding="utf-8")

        return catalog_df


# ---------------------------------------------------------------------
# (2) Shallow CSV dialect/structure inspector
# ---------------------------------------------------------------------
@dataclass
class DialectInspectorConfig:
    """
    Shallow (fast) inspection parameters.

    - inspect_rows: maximum number of data rows to scan (streaming).
    - delimiter: preferred delimiter; if None, attempt `csv.Sniffer` first.
    - encoding_candidates: attempted encodings for robust reads.
    """
    inspect_rows: int = 200
    delimiter: Optional[str] = None
    encoding_candidates: Tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")


class CSVDialectInspector:
    """
    Inspect CSV structural consistency (header vs rows).

    Output columns appended to the catalog:
      - csv_header_ncols
      - csv_mode_ncols
      - csv_bad_rows_count
      - csv_extra_field_candidate
      - csv_extra_field_constant_value
      - csv_read_ok
      - csv_dialect_ok
      - csv_delimiter (extra, non-breaking)
      - csv_encoding (extra, non-breaking)

    The "extra field candidate" detection flags the common pattern:
      mode_ncols == header_ncols + 1 and the last field is (almost) constant.

    This inspector is intentionally shallow: it does not validate numeric ranges
    or load the full dataset.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        output_report_path: Optional[Path] = None,
        *,
        inspector_cfg: Optional[DialectInspectorConfig] = None,
    ) -> None:
        self.exp_cfg = cfg
        self.cfg = inspector_cfg or DialectInspectorConfig()
        self.output_report_path = Path(output_report_path) if output_report_path else (cfg.target_root / "dialect_report.json")

    def _open_text(self, path: Path):
        last_err: Optional[Exception] = None
        for enc in self.cfg.encoding_candidates:
            try:
                return path.open("r", encoding=enc, newline="")
            except Exception as e:
                last_err = e
        raise last_err or OSError(f"Unable to open file: {path}")

    def _sniff_delimiter(self, sample: str) -> Optional[str]:
        # Try to sniff the delimiter from a small sample.
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except Exception:
            return None

    def _inspect_one(self, path: Path) -> Dict[str, object]:
        """
        Shallow inspection of a CSV file.

        Returns keys:
          read_ok, reason, delimiter, encoding,
          header_ncols, mode_ncols, bad_rows_count,
          extra_field_candidate, extra_field_constant_value, examples_extra
        """
        inspect_rows = self.cfg.inspect_rows

        # Choose delimiter: preferred -> sniff -> fallback comma
        delimiter = self.cfg.delimiter

        # Read a small sample for sniffing (doesn't load whole file)
        enc_used = None
        sample = ""
        with self._open_text(path) as f:
            enc_used = getattr(f, "encoding", None)
            sample = f.read(4096)
        if delimiter is None:
            delimiter = self._sniff_delimiter(sample) or ","

        # Now re-open to stream rows from the beginning
        with self._open_text(path) as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                return {
                    "read_ok": False,
                    "reason": "empty_file",
                    "delimiter": delimiter,
                    "encoding": enc_used,
                    "header_ncols": None,
                    "mode_ncols": None,
                    "bad_rows_count": None,
                    "extra_field_candidate": False,
                    "extra_field_constant_value": None,
                    "examples_extra": [],
                }

            header_n = len(header)
            counts = Counter()
            bad_rows = 0
            extra_values: List[str] = []

            for i, row in enumerate(reader):
                if i >= inspect_rows:
                    break
                n = len(row)
                counts[n] += 1
                if n != header_n:
                    bad_rows += 1
                    if n == header_n + 1 and len(extra_values) < 10:
                        extra_values.append(row[-1])

            mode_n = counts.most_common(1)[0][0] if counts else None

            # Detect candidate "extra field" that is (almost) constant.
            extra_candidate = False
            const_val = None
            if mode_n is not None and mode_n == header_n + 1 and extra_values:
                c = Counter(extra_values)
                top_val, top_count = c.most_common(1)[0]
                if top_count >= max(3, int(0.8 * len(extra_values))):
                    extra_candidate = True
                    const_val = top_val

            return {
                "read_ok": True,
                "reason": "",
                "delimiter": delimiter,
                "encoding": enc_used,
                "header_ncols": header_n,
                "mode_ncols": mode_n,
                "bad_rows_count": bad_rows,
                "extra_field_candidate": extra_candidate,
                "extra_field_constant_value": const_val,
                "examples_extra": extra_values[:5],
            }

    def run(
        self,
        catalog_df: pd.DataFrame,
        *,
        inspect_rows: Optional[int] = None,
        write: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Run inspection for all existing CSVs referenced by the catalog.

        Returns:
          (catalog_df_augmented, report_dict)
        """
        if inspect_rows is not None:
            self.cfg.inspect_rows = int(inspect_rows)

        required = {"mid", "raw_path", "raw_exists"}
        if not required.issubset(set(catalog_df.columns)):
            raise ValueError(f"catalog_df must contain columns: {sorted(required)}")

        df = catalog_df.copy()

        header_ncols_col: List[Optional[int]] = []
        mode_ncols_col: List[Optional[int]] = []
        bad_rows_col: List[Optional[int]] = []
        extra_candidate_col: List[bool] = []
        extra_const_col: List[Optional[str]] = []
        read_ok_col: List[bool] = []
        dialect_ok_col: List[bool] = []
        delimiter_col: List[Optional[str]] = []
        encoding_col: List[Optional[str]] = []

        per_file: Dict[str, Dict[str, object]] = {}
        summary_counts = Counter()

        for _, r in df.iterrows():
            mid = _safe_str(r["mid"])
            p = Path(_safe_str(r["raw_path"]))
            exists = bool(r["raw_exists"])

            if not exists:
                stats = {
                    "read_ok": False,
                    "reason": "missing_file",
                    "delimiter": None,
                    "encoding": None,
                    "header_ncols": None,
                    "mode_ncols": None,
                    "bad_rows_count": None,
                    "extra_field_candidate": False,
                    "extra_field_constant_value": None,
                    "examples_extra": [],
                }
                dialect_ok = False
                summary_counts["missing"] += 1
            else:
                try:
                    stats = self._inspect_one(p)
                    dialect_ok = bool(stats["read_ok"]) and int(stats["bad_rows_count"] or 0) == 0
                    summary_counts["inspected"] += 1
                    if dialect_ok:
                        summary_counts["dialect_ok"] += 1
                    if stats.get("extra_field_candidate"):
                        summary_counts["extra_field_candidate"] += 1
                except Exception as e:
                    stats = {
                        "read_ok": False,
                        "reason": f"read_error: {type(e).__name__}",
                        "delimiter": None,
                        "encoding": None,
                        "header_ncols": None,
                        "mode_ncols": None,
                        "bad_rows_count": None,
                        "extra_field_candidate": False,
                        "extra_field_constant_value": None,
                        "examples_extra": [],
                    }
                    dialect_ok = False
                    summary_counts["errors"] += 1

            per_file[mid] = stats

            header_ncols_col.append(stats.get("header_ncols"))
            mode_ncols_col.append(stats.get("mode_ncols"))
            bad_rows_col.append(stats.get("bad_rows_count"))
            extra_candidate_col.append(bool(stats.get("extra_field_candidate", False)))
            extra_const_col.append(stats.get("extra_field_constant_value"))
            read_ok_col.append(bool(stats.get("read_ok", False)))
            dialect_ok_col.append(bool(dialect_ok))
            delimiter_col.append(stats.get("delimiter"))
            encoding_col.append(stats.get("encoding"))

        df["csv_header_ncols"] = header_ncols_col
        df["csv_mode_ncols"] = mode_ncols_col
        df["csv_bad_rows_count"] = bad_rows_col
        df["csv_extra_field_candidate"] = extra_candidate_col
        df["csv_extra_field_constant_value"] = extra_const_col
        df["csv_read_ok"] = read_ok_col
        df["csv_dialect_ok"] = dialect_ok_col
        # New non-breaking columns
        df["csv_delimiter"] = delimiter_col
        df["csv_encoding"] = encoding_col

        report = {
            "inspector": {
                "inspect_rows": self.cfg.inspect_rows,
                "preferred_delimiter": self.cfg.delimiter,
                "encoding_candidates": list(self.cfg.encoding_candidates),
            },
            "summary": dict(summary_counts),
            "per_file": per_file,
        }

        if write:
            out = self.output_report_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        return df, report
