# informational_config.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal
import json
import re

import pandas as pd

# =============================================================================
# Date helpers: parse ddMmmYY in Spanish tokens (08Ene25, 10Dic24, etc.)
# =============================================================================

_MONTHS_ES: Dict[str, int] = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}


def _parse_fecha_ddMmmYY(token: str) -> Tuple[int, int, int]:
    m = re.fullmatch(r"(\d{2})([A-Za-z]{3})(\d{2})", str(token).strip())
    if not m:
        raise ValueError(f"Invalid date token: {token!r}. Expected ddMmmYY (e.g. '08Ene25').")
    dd = int(m.group(1))
    mmm = m.group(2).capitalize()
    yy = int(m.group(3))
    if mmm not in _MONTHS_ES:
        raise ValueError(f"Invalid month in date token: {token!r} (month={mmm!r}).")
    mm = _MONTHS_ES[mmm]
    yyyy = 2000 + yy
    return (yyyy, mm, dd)


def _sort_fechas(tokens: Sequence[str]) -> List[str]:
    return sorted([str(x) for x in tokens], key=_parse_fecha_ddMmmYY)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CSV helpers (low-resource friendly)
# =============================================================================

_ENCODING_CANDIDATES: Tuple[str, ...] = ("utf-8", "latin-1")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns (strip, remove BOM)
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def _read_csv_header(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in _ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(path, nrows=0, encoding=enc)
            return _clean_columns(df)
        except Exception as e:
            last_err = e
    raise ValueError(f"Failed to read CSV header: {path} ({type(last_err).__name__}: {last_err})")


def _read_csv_usecols(
    path: Path,
    *,
    usecols: Sequence[str],
    nrows: Optional[int] = None,
    dtype: Any = "string",
) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in _ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(path, usecols=list(usecols), nrows=nrows, dtype=dtype, encoding=enc)
            return _clean_columns(df)
        except Exception as e:
            last_err = e
    raise ValueError(f"Failed to read CSV: {path} ({type(last_err).__name__}: {last_err})")



def _read_csv_small(path: Path, nrows: int = 50) -> pd.DataFrame:
    # Useful for diagnostics
    last_err: Optional[Exception] = None
    for enc in _ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(path, nrows=nrows, encoding=enc)
            return _clean_columns(df)
        except Exception as e:
            last_err = e
    raise ValueError(f"Failed to read CSV: {path} ({type(last_err).__name__}: {last_err})")


def _require_cols(columns: Sequence[str], required: Sequence[str], ctx: str) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {ctx}. "
            f"Available columns: {list(columns)}"
        )


def _normalize_string_values(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if not s:
            continue
        low = s.lower()
        if low in {"nan", "none", "null"}:
            continue
        out.append(s)
    return out


def _unique_nonempty_from_series(s: pd.Series) -> List[str]:
    vals = _normalize_string_values(s.dropna().tolist())
    return sorted(set(vals))


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    # Works even if column was read as string
    if s.dtype == bool:
        return s
    ss = s.astype("string").fillna("").str.strip().str.lower()
    return ss.isin({"1", "true", "t", "yes", "y"})


def _sample_existing_files(paths: Sequence[str], sample_n: int = 25) -> Tuple[int, int, List[str]]:
    checked = 0
    missing = 0
    examples: List[str] = []
    for p in paths[: max(0, int(sample_n))]:
        checked += 1
        pp = Path(str(p))
        if not pp.exists():
            missing += 1
            if len(examples) < 5:
                examples.append(str(pp))
    return checked, missing, examples


# =============================================================================
# Declarative Level 4 configuration spec
# =============================================================================

MetricName = Literal["KL", "JS"]
MultiCompCorrection = Literal["none", "bonferroni", "bh_fdr", "dunnett_like"]
ExperimentMode = Literal["blind", "declared"]

@dataclass(frozen=True)
class QueuePaths:
    quality_scores_by_file: Path
    resultados_luces: Path

@dataclass(frozen=True)
class QueueSchema:
    required_cols: Tuple[str, ...] = (
        "mid",
        "fecha",
        "lab",
        "archivo",
        "parquet_path",
    )

@dataclass(frozen=True)
class PairingKeys:
    across_lab_same_file: Tuple[str, ...] = ("fecha", "archivo")
    within_date_lab: Tuple[str, ...] = ("fecha", "lab")

@dataclass(frozen=True)
class BaselineAssociation:
    enabled: bool = False
    note: str = "Legacy baseline association disabled under active multi-lab contract."

@dataclass(frozen=True)
class CouplingConfig:
    enabled: bool = True
    lags_samples: Tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class DecisionConfig:
    alpha: float = 0.05
    metric_primary: MetricName = "JS"
    metric_secondary: MetricName = "KL"
    correction: MultiCompCorrection = "none"


@dataclass(frozen=True)
class EffectSizeConfig:
    bootstrap: bool = True
    n_boot: int = 500
    bootstrap_mode: Literal["block"] = "block"
    block_size_hint: int = 500




@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    mode: ExperimentMode
    errors: Tuple[str, ...] = tuple()
    warnings: Tuple[str, ...] = tuple()
    summary: Dict[str, Any] = field(default_factory=dict)


class InformationalConfigError(RuntimeError):
    pass


@dataclass
class InformationalConfig:
    target_root: Path
    analysis_ready_dir: Path
    level2_reports_dir: Path
    queues: QueuePaths

    mode: ExperimentMode = "blind"

    queue_schema: QueueSchema = field(default_factory=QueueSchema)
    pairing_keys: PairingKeys = field(default_factory=PairingKeys)
    baseline_assoc: BaselineAssociation = field(default_factory=BaselineAssociation)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)

    decision: DecisionConfig = field(default_factory=DecisionConfig)
    effect_size: EffectSizeConfig = field(default_factory=EffectSizeConfig)

    lasers: Tuple[str, ...] = tuple()
    lasers_source: str = "unset"  # NEW: audit/debug

    bins_reference_group: Literal["ALL", "UNK"] = "ALL"
    bins_reference_dates: Optional[Tuple[str, ...]] = None

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    @classmethod
    def from_experiment(
        cls,
        cfg: Any,
        *,
        quality_scores_by_file_path: Path,
        resultados_luces_path: Path,
        level4_folder_name: str = "Level4_Informational",
        metric_primary: MetricName = "JS",
        alpha: float = 0.05,
        correction: MultiCompCorrection = "none",
        coupling_lags_samples: Tuple[int, ...] = (0, 1),
    ) -> "InformationalConfig":
        target_root = Path(getattr(cfg, "target_root"))
        analysis_ready_dir = target_root / "Analysis Ready"
        level2_reports_dir = target_root / "Reports" / level4_folder_name

        inst = cls(
            target_root=target_root,
            analysis_ready_dir=analysis_ready_dir,
            level2_reports_dir=level2_reports_dir,
            queues=QueuePaths(
                quality_scores_by_file=Path(quality_scores_by_file_path),
                resultados_luces=Path(resultados_luces_path),
            ),
            mode="blind",
            decision=DecisionConfig(
                alpha=float(alpha),
                metric_primary=metric_primary,
                correction=correction,
            ),
            coupling=CouplingConfig(
                enabled=True,
                lags_samples=tuple(int(x) for x in coupling_lags_samples),
            ),
            bins_reference_group="ALL",
        )










        # Load lasers from quality outputs under the active multi-lab contract
        lasers, src = inst._load_lasers_from_quality_outputs()

        # WARNING: TEMPORARY FILTER. This is intentionally pragmatic to make the pipeline run.
        # IT MUST BE REPLACED BY A SCHEMA-DRIVEN SENSOR TAXONOMY (LASERS vs IMU) FOR SCALABILITY.
        lasers, dropped = inst._filter_lasers_with_data(list(lasers))

        inst.lasers = tuple(lasers)
        inst.lasers_source = src + ((" | dropped_nonlaser=" + ",".join(dropped)) if dropped else "")

        # DECLARED mode: cohorts + default contrasts are available


        inst.validate(strict=True)
        return inst

    def _filter_lasers_with_data(self, lasers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Filter out non-laser channels.

        Motivation:
          - Some runs include non-laser sensors in the channel list (e.g., AccelZ -> Ch7).
          - Informational bins/states (amplitude/energy/fourier/movement) are defined for laser channels only.

        Strategy (TEMPORARY):
          1) Prefer column_roles.json if present to identify raw names like 'Luz1..' as laser channels.
          2) Fallback to a conservative regex filter if the roles file is missing/unreadable.

        WARNING: TEMPORARY FIX — MUST BE REPLACED BY SCHEMA-DRIVEN DISCOVERY FOR SCALABILITY.
        """
        valid: List[str] = []
        dropped: List[str] = []

        roles_path = self.target_root / "column_roles.json"
        laser_like: Optional[set] = None

        if roles_path.exists():
            try:
                obj = json.loads(roles_path.read_text(encoding="utf-8"))
                s: set = set()
                # obj is {mid -> {roles -> {channels: [{raw_name, canonical_name}, ...]}}}
                for rec in obj.values():
                    roles = rec.get("roles", {}) if isinstance(rec, dict) else {}
                    chans = roles.get("channels", []) if isinstance(roles, dict) else []
                    if not isinstance(chans, list):
                        continue
                    for c in chans:
                        if not isinstance(c, dict):
                            continue
                        raw_name = str(c.get("raw_name", "")).strip().lower()
                        canon = str(c.get("canonical_name", "")).strip()
                        if not canon:
                            continue
                        # Heuristic: Spanish 'Luz*' are laser channels
                        if raw_name.startswith("luz") or ("laser" in raw_name) or ("light" in raw_name):
                            s.add(canon)
                if s:
                    laser_like = s
            except Exception as e:
                print(f"[Level2][WARN] Could not parse column_roles.json: {roles_path} | {e}")

        for l in lasers:
            ss = str(l).strip()
            if not ss:
                continue
            if laser_like is not None:
                if ss in laser_like:
                    valid.append(ss)
                else:
                    dropped.append(ss)
            else:
                # Conservative fallback: keep only Ch1..Ch6 (typical laser set); drop others (e.g., Ch7/AccelZ).
                if re.fullmatch(r"Ch[1-6]", ss):
                    valid.append(ss)
                else:
                    dropped.append(ss)

        if dropped:
            print("[Level2][WARN] Dropping non-laser channels:", dropped)

        return valid, dropped

    def _load_lasers_from_quality_outputs(self) -> Tuple[List[str], str]:
        """
        Load laser/channel names from resultados_luces.csv.
        Active contract: derive informational channels from Quality outputs,
        not from legacy stability artifacts.
        """
        src = Path(self.queues.resultados_luces)
        if not src.exists():
            raise FileNotFoundError(f"resultados_luces not found: {src}")

        header = _read_csv_header(src)
        available = set(header.columns)

        laser_col = None
        for candidate in ("laser", "laser_name"):
            if candidate in available:
                laser_col = candidate
                break

        if laser_col is None:
            raise ValueError(
                f"Missing required laser column in resultados_luces: {src}. "
                f"Expected one of ['laser', 'laser_name']. "
                f"Available columns: {list(header.columns)}"
            )

        df = _read_csv_usecols(src, usecols=[laser_col], dtype="string")
        lasers = _unique_nonempty_from_series(df[laser_col])

        if not lasers:
            raise ValueError(
                f"No non-empty laser values found in resultados_luces using column '{laser_col}': {src}"
            )

        return lasers, f"quality_output:{src.name}:{laser_col}"
    
    # -------------------------------------------------------------------------
    # Validation + preflight
    # -------------------------------------------------------------------------
    def validate(self, *, strict: bool = True) -> None:
        if not (0.0 < float(self.decision.alpha) < 1.0):
            raise ValueError(f"Invalid alpha: {self.decision.alpha}. Must be in (0,1).")

        if not self.lasers:
            raise ValueError("Empty lasers list. Check resultados_luces and laser extraction.")
        if strict:
            if not self.analysis_ready_dir.exists(): 
                raise FileNotFoundError(f"Analysis Ready directory not found: {self.analysis_ready_dir}")
            #
            if not Path(self.queues.quality_scores_by_file).exists():
                raise FileNotFoundError(
                    f"quality_scores_by_file not found: {self.queues.quality_scores_by_file}"
                )
            if not Path(self.queues.resultados_luces).exists():
                raise FileNotFoundError(
                    f"resultados_luces not found: {self.queues.resultados_luces}"
                )

            hdr = _read_csv_header(Path(self.queues.quality_scores_by_file))
            _require_cols(
                hdr.columns,
                self.queue_schema.required_cols,
                ctx=f"quality_scores_by_file={self.queues.quality_scores_by_file}",
            )

            df_sample = _read_csv_usecols(
                Path(self.queues.quality_scores_by_file),
                usecols=["archivo"],
                nrows=200,
                dtype="string",
            )         
         
         
            vals = df_sample["archivo"].dropna().astype("string").head(50).tolist()
            for s in vals:
                ss = str(s).strip()
                if ss and not re.fullmatch(r"\d+", ss):
                    raise ValueError(
                        "Column 'archivo' must be numeric (int or numeric string). "
                        f"Found invalid example: {ss!r}"
                    )


    def preflight(
        self,
        *,
        strict: bool = True,
        parquet_sample_n: int = 25,
        verbose: bool = True,
    ) -> PreflightReport:
        errors: List[str] = []
        warnings: List[str] = []

        try:
            self.validate(strict=strict)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")

        summary: Dict[str, Any] = {
            "target_root": str(self.target_root),
            "analysis_ready_dir": str(self.analysis_ready_dir),
            "level2_reports_dir": str(self.level2_reports_dir),
            "quality_scores_by_file": str(self.queues.quality_scores_by_file),
            "resultados_luces": str(self.queues.resultados_luces),
            "mode": self.mode,
            "n_lasers": len(self.lasers),
            "lasers": list(self.lasers),
            "lasers_source": self.lasers_source,
            "alpha": float(self.decision.alpha),
            "metric_primary": self.decision.metric_primary,
        }

        # Quick stats: quality queue
        # Quick stats: quality queue
        if Path(self.queues.quality_scores_by_file).exists():
            try:
                cols = ["fecha", "lab", "parquet_path"]
                hdr = _read_csv_header(Path(self.queues.quality_scores_by_file))
                available = set(hdr.columns)
                cols = [c for c in cols if c in available]
                if cols:
                    df = _read_csv_usecols(
                        Path(self.queues.quality_scores_by_file),
                        usecols=cols,
                        dtype="string",
                    )
                    if "fecha" in df.columns:
                        summary["n_dates_in_queue"] = int(df["fecha"].dropna().nunique())
                    if "lab" in df.columns:
                        summary["labs_in_queue"] = sorted(
                            _unique_nonempty_from_series(df["lab"])
                        )

                    if "parquet_path" in df.columns and parquet_sample_n > 0:
                        pq_paths = df["parquet_path"].dropna().astype("string").tolist()
                        checked, missing, examples = _sample_existing_files(
                            pq_paths,
                            sample_n=parquet_sample_n,
                        )
                        summary["parquet_sample_checked"] = checked
                        summary["parquet_sample_missing"] = missing
                        if missing > 0:
                            warnings.append(
                                f"Some parquet_path entries are missing on disk (sample): {missing}/{checked}. "
                                f"Examples: {examples}"
                            )
            except Exception as e:
                warnings.append(
                    f"Quality queue quick-stats failed: {type(e).__name__}: {e}"
                )
                
        if self.mode == "blind":
            warnings.append(
                "BLIND mode active under current multi-lab contract. "
                "Informational modules must operate on fecha -> lab -> medición without Ctrl/Exp assumptions."
            )
            summary["bins_reference_group_recommended"] = self.bins_reference_group
            
        ok = (len(errors) == 0)
        report = PreflightReport(
            ok=ok,
            mode=self.mode,
            errors=tuple(errors),
            warnings=tuple(warnings),
            summary=summary,
        )

        if verbose:
            self._print_preflight(report)

        if strict and not report.ok:
            raise InformationalConfigError(
                "Informational Level 4 preflight failed:\n- " + "\n- ".join(report.errors)
            )
        return report

    @staticmethod
    def _print_preflight(report: PreflightReport) -> None:
        print("=" * 72)
        print("[Level4][InformationalConfig] PREFLIGHT")
        print("=" * 72)
        print(f"Mode: {report.mode}")
        print(f"OK:   {report.ok}")
        print("-" * 72)

        s = report.summary
        print(f"target_root:       {s.get('target_root')}")
        print(f"analysis_ready:    {s.get('analysis_ready_dir')}")
        print(f"level4_reports:    {s.get('level2_reports_dir')}")
        print(f"quality_scores:    {s.get('quality_scores_by_file')}")
        print(f"resultados_luces:  {s.get('resultados_luces')}")
        print(f"lasers:            {s.get('n_lasers')} -> {s.get('lasers')}")
        print(f"lasers_source:     {s.get('lasers_source')}")
        if "n_dates_in_queue" in s:
            print(f"dates_in_queue:    {s.get('n_dates_in_queue')}")
        if "parquet_sample_checked" in s:
            print(f"parquet_sample:    checked={s.get('parquet_sample_checked')} missing={s.get('parquet_sample_missing')}")

        if report.errors:
            print("-" * 72)
            print("ERRORS:")
            for e in report.errors:
                print(f"  - {e}")

        if report.warnings:
            print("-" * 72)
            print("WARNINGS:")
            for w in report.warnings:
                print(f"  - {w}")

        print("=" * 72)

    # -------------------------------------------------------------------------
    # Serialization / auditing
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)

        def convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        return convert(d)

    def write_json(self, path: Path) -> Path:
        _ensure_dir(Path(path).parent)
        payload = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return Path(path)

    # -------------------------------------------------------------------------
    # Minimal API for downstream modules
    # -------------------------------------------------------------------------
    def get_level4_paths(self) -> Dict[str, Path]:
        return {
            "analysis_ready_dir": self.analysis_ready_dir,
            "level4_reports_dir": self.level2_reports_dir,
            "quality_scores_by_file": self.queues.quality_scores_by_file,
            "resultados_luces": self.queues.resultados_luces,
            "level4_config_json": self.level2_reports_dir / "level4_config.json",
            "bins_spec_json": self.level2_reports_dir / "bins_spec.json",
            "states_parquet": self.level2_reports_dir / "states_by_mid.parquet",
            "effects_csv": self.level2_reports_dir / "effects_table.csv",
            "decisions_csv": self.level2_reports_dir / "decisions_table.csv",
            "hysteresis_csv": self.level2_reports_dir / "hysteresis_table.csv",
            "tex_payload_json": self.level2_reports_dir / "tex_payload.json",
        }
        
__all__ = [
    "MetricName",
    "MultiCompCorrection",
    "ExperimentMode",
    "QueuePaths",
    "QueueSchema",
    "PairingKeys",
    "BaselineAssociation",
    "CouplingConfig",
    "DecisionConfig",
    "EffectSizeConfig",
    "PreflightReport",
    "InformationalConfigError",
    "InformationalConfig",
]
