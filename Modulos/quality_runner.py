from __future__ import annotations

"""
quality_runner.py (Level 1A: Quality) - ORCHESTRATOR

Single responsibility:
- Iterate over "Analysis Ready" canonical Parquet tables,
- compute per-channel metrics/validations using quality_metrics.py,
- write output artifacts (CSV/TXT) for downstream pipeline stages.

This module does NOT:
- perform informational inference (Level 2),
- generate comparative figures (those belong elsewhere).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from quality_metrics import calcular_metricas_luz, validar_valores_luz

try:  # Optional fast schema access (no data load)
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None


# -----------------------------------------------------------------------------
# Public artifacts
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityRunArtifacts:
    output_dir: Path
    resultados_luces_csv: Path
    datos_completos_luces_csv: Path
    scores_by_file_csv: Path
    summary_by_lab_csv: Path
    summary_by_date_lab_csv: Path
    summary_by_laser_csv: Path
    resumen_ejecutivo_txt: Path


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

class QualityRunner:
    """
    Runs Level 1A (Quality) on Analysis Ready Parquet files.

    Expected layout (v3 default):
    <root>/Analysis Ready/<fecha>/<lab>/<mid>.parquet

    Legacy-compatible optional layout:
    <root>/Analysis Ready/<fecha>/<lab>/<jornada>/<mid>.parquet
    
    Inputs:
      - catalog_df: augmented catalog DataFrame (from analysis_ready_prep)
      - schema_by_file_csv (optional): output of analysis_ready_schema_table,
        used to skip non-comparable files (schema_ok == False).

    Outputs (in output_dir):
      - resultados_luces.csv       (compact table per channel/per file)
      - datos_completos_luces.csv  (full table with all computed fields)
      - quality_scores_by_file.csv (aggregated score per file/mid)
      - resumen_ejecutivo.txt      (brief summary)
    """

    def __init__(
        self,
        *,
        root: Path,
        config: Any,  # QualityConfig
        analysis_ready_dir_name: str = "Analysis Ready",
        default_output_dir: str = "Reports/Level1_Quality",
    ) -> None:
        self.root = Path(root)
        self.cfg = config
        self.analysis_root = self.root / analysis_ready_dir_name
        self.default_output_dir = default_output_dir

    # ---------------------------------------------------------------------

    def run(
        self,
        *,
        catalog_df: pd.DataFrame,
        schema_by_file_csv: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        write: bool = True,
        verbose: bool = True,
    ) -> QualityRunArtifacts:
        # ---- Minimal input validation
        required = {"mid", "fecha", "lab"}
        missing = [c for c in required if c not in catalog_df.columns]
        if missing:
            raise ValueError(f"catalog_df is missing required columns: {missing}")

        if not self.analysis_root.exists():
            raise FileNotFoundError(f"Analysis Ready directory does not exist: {self.analysis_root}")

        # ---- Output dir
        out_dir = Path(output_dir) if output_dir else (self.root / self.default_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        #
        
        artifacts_cfg = getattr(self.cfg, "artifacts", None)
        resultados_name = getattr(artifacts_cfg, "resultados_luces_csv", "resultados_luces.csv")
        completos_name = getattr(artifacts_cfg, "datos_completos_luces_csv", "datos_completos_luces.csv")
        scores_name = getattr(artifacts_cfg, "quality_scores_by_file_csv", "quality_scores_by_file.csv")
        summary_lab_name = getattr(artifacts_cfg, "quality_summary_by_lab_csv", "quality_summary_by_lab.csv")
        summary_date_lab_name = getattr(artifacts_cfg, "quality_summary_by_date_lab_csv", "quality_summary_by_date_lab.csv")
        summary_laser_name = getattr(artifacts_cfg, "quality_summary_by_laser_csv", "quality_summary_by_laser.csv")
        resumen_name = getattr(artifacts_cfg, "resumen_ejecutivo_txt", "resumen_ejecutivo.txt")

        resultados_csv = out_dir / str(resultados_name)
        completos_csv = out_dir / str(completos_name)
        scores_csv = out_dir / str(scores_name)
        summary_by_lab_csv = out_dir / str(summary_lab_name)
        summary_by_date_lab_csv = out_dir / str(summary_date_lab_name)
        summary_by_laser_csv = out_dir / str(summary_laser_name)
        resumen_txt = out_dir / str(resumen_name)
        
        # ---- Schema gate (optional)
        schema_ok_by_mid = self._load_schema_ok_map(schema_by_file_csv)

        # ---- Channel policy
        desired_channels = self._desired_channels()
        allowed_prefixes = tuple(getattr(self.cfg.channels, "allowed_prefixes", ("Luz",)))

        # ---- Accumulators
        rows_full: List[Dict[str, Any]] = []
        rows_scores: List[Dict[str, Any]] = []
        skipped: List[Tuple[str, str, str]] = []  # (mid, parquet_path, reason)
        #
        
        min_valid_samples = int(getattr(self.cfg.cleaning, "min_valid_samples", 10))
        min_channels_required = int(getattr(self.cfg.channels, "min_channels_required", 4))
        pass_threshold = float(getattr(self.cfg.thresholds, "pass_to_informational", 7.0))
        
        # ---- Iterate catalog rows (fast path)
        for row in catalog_df.itertuples(index=False):
            meta = self._extract_meta(row)

            mid = meta["mid"]

            # 0) Schema gate (if provided)
            if schema_ok_by_mid and not bool(schema_ok_by_mid.get(mid, False)):
                parquet_path = self._resolve_parquet_path(meta)
                skipped.append((mid, str(parquet_path), "schema_not_ok"))
                continue

            # 1) Resolve Parquet path (supports mid-based and legacy naming)
            parquet_path = self._resolve_parquet_path(meta)
            parquet_path_str = str(parquet_path)

            if not parquet_path.exists():
                skipped.append((mid, parquet_path_str, "parquet_missing"))
                continue

            # 2) Determine channel columns WITHOUT loading full table when possible
            channel_cols = self._detect_channels(
                parquet_path=parquet_path,
                desired_channels=desired_channels,
                allowed_prefixes=allowed_prefixes,
            )
            if not channel_cols:
                skipped.append((mid, parquet_path_str, "no_channel_columns"))
                continue

            # 3) Read only channel columns (memory saver)
            try:
                df = pd.read_parquet(parquet_path, columns=channel_cols)
            except Exception:
                skipped.append((mid, parquet_path_str, "parquet_read_error"))
                continue

            # 4) Per-channel metrics
            per_file_scores: List[float] = []
            n_valid_channels = 0
            n_total_channels = int(len(channel_cols))

            for ch in channel_cols:
                series = self._clean_series(df, ch)
                if series is None:
                    #
                    rows_full.append(self._row_base(meta, parquet_path_str, ch, status="missing_col"))
                    continue

                if len(series) < min_valid_samples:
                    rows_full.append(
                        self._row_base(
                            meta,
                            parquet_path_str,
                            ch,
                            status=f"insufficient_samples<{min_valid_samples}",
                        )
                    )
                    continue

                # Minimal DF to avoid NaN/Inf strictness in validar_valores_luz
                df_ch = pd.DataFrame({ch: series.to_numpy(copy=False)})

                metrics = calcular_metricas_luz(
                    df_luz=df_ch,
                    columna=ch,
                    nombre_archivo=mid,
                    cfg=self.cfg,
                )
                if metrics is None:
                    rows_full.append(self._row_base(meta, parquet_path_str, ch, status="metrics_none"))
                    continue

                validation = validar_valores_luz(df_ch, ch, cfg=self.cfg)
                is_valid = bool(validation.get("valido", False))

                # Avoid key collisions with our metadata ("archivo" in metrics is a label string)
                metric_source_id = str(metrics.pop("archivo", ""))
                metrics.pop("columna", None)
                #
                
                rec: Dict[str, Any] = {
                    **meta,
                    "parquet_path": parquet_path_str,

                    # --- Laser identity ---
                    "laser_name": ch,
                    "columna": ch,

                    "valido": is_valid,
                    "razon_validacion": str(validation.get("razon", "")),
                    "metric_source_id": metric_source_id,
                    **metrics,
                    "score_canal": float(metrics.get("calidad_general", 0.0)),
                    "channel_detected": True,
                    "status": "ok" if is_valid else "validation_failed",
                }

                rows_full.append(rec)

                if is_valid:
                    per_file_scores.append(float(rec.get("calidad_general", 0.0)))
                    n_valid_channels += 1

            # 5) Aggregate file score
            score_file = float(sum(per_file_scores) / len(per_file_scores)) if per_file_scores else 0.0
            has_min_channels = n_valid_channels >= min_channels_required
            pass_to_level2 = bool(has_min_channels and (score_file >= pass_threshold))

            
            rows_scores.append(
                {
                    **{k: meta[k] for k in meta.keys()},
                    "parquet_path": parquet_path_str,
                    "n_channels_total": n_total_channels,
                    "n_channels_used": int(n_valid_channels),
                    "n_channels_invalid": int(n_total_channels - n_valid_channels),
                    "min_channels_required": int(min_channels_required),
                    "score_medicion": score_file,
                    "score_mean_valid_channels": score_file,
                    "pass_to_level2": pass_to_level2,
                    "threshold_pass": pass_threshold,
                    "status": "ok" if has_min_channels else "insufficient_valid_channels",
                }
            )

            # Free memory aggressively for low-resource machines
            del df

        # ---- DataFrames
        df_full = pd.DataFrame(rows_full)
        df_scores = pd.DataFrame(rows_scores)
        df_skipped = (
            pd.DataFrame(skipped, columns=["mid", "parquet_path", "skip_reason"])
            if skipped
            else pd.DataFrame(columns=["mid", "parquet_path", "skip_reason"])
        )

        # A compact subset for downstream (gate/compare)
        
        keep_cols = [
            "mid",
            "fecha",
            "lab",
            "turno",
            "jornada",
            "etiqueta",
            "archivo",
            "color",
            "parquet_path",
            "laser_name",
            "columna",
            "channel_detected",
            "valido",
            "razon_validacion",
            "media",
            "desviacion",
            "ruido_rms",
            "snr_db",
            "coef_variacion",
            "tendencia_por_muestra",
            "valores_fuera_rango",
            "porcentaje_fuera_rango",
            "calidad_general",
            "score_canal",
            "num_muestras",
            "muestras_suficientes",
            "status",
        ]
        
        df_resultados = df_full[[c for c in keep_cols if c in df_full.columns]].copy()
        # ---- Multi-lab aggregated summaries
        df_valid = df_full.copy()
        if "valido" in df_valid.columns:
            df_valid = df_valid[df_valid["valido"] == True].copy()

        if not df_valid.empty:
            df_summary_by_lab = (
                df_valid.groupby(["lab"], dropna=False)
                .agg(
                    n_registros=("mid", "count"),
                    n_mids=("mid", "nunique"),
                    n_lasers=("laser_name", "nunique"),
                    score_promedio=("calidad_general", "mean"),
                    score_std=("calidad_general", "std"),
                    snr_promedio=("snr_db", "mean"),
                    cv_promedio=("coef_variacion", "mean"),
                    pct_fuera_rango_promedio=("porcentaje_fuera_rango", "mean"),
                )
                .reset_index()
            )

            df_summary_by_date_lab = (
                df_valid.groupby(["fecha", "lab"], dropna=False)
                .agg(
                    n_registros=("mid", "count"),
                    n_mids=("mid", "nunique"),
                    n_lasers=("laser_name", "nunique"),
                    score_promedio=("calidad_general", "mean"),
                    score_std=("calidad_general", "std"),
                    snr_promedio=("snr_db", "mean"),
                    cv_promedio=("coef_variacion", "mean"),
                    pct_fuera_rango_promedio=("porcentaje_fuera_rango", "mean"),
                )
                .reset_index()
            )

            df_summary_by_laser = (
                df_valid.groupby(["lab", "laser_name"], dropna=False)
                .agg(
                    n_registros=("mid", "count"),
                    n_mids=("mid", "nunique"),
                    score_promedio=("calidad_general", "mean"),
                    score_std=("calidad_general", "std"),
                    snr_promedio=("snr_db", "mean"),
                    cv_promedio=("coef_variacion", "mean"),
                    pct_fuera_rango_promedio=("porcentaje_fuera_rango", "mean"),
                )
                .reset_index()
            )
        else:
            df_summary_by_lab = pd.DataFrame(
                columns=[
                    "lab", "n_registros", "n_mids", "n_lasers",
                    "score_promedio", "score_std", "snr_promedio",
                    "cv_promedio", "pct_fuera_rango_promedio"
                ]
            )
            df_summary_by_date_lab = pd.DataFrame(
                columns=[
                    "fecha", "lab", "n_registros", "n_mids", "n_lasers",
                    "score_promedio", "score_std", "snr_promedio",
                    "cv_promedio", "pct_fuera_rango_promedio"
                ]
            )
            df_summary_by_laser = pd.DataFrame(
                columns=[
                    "lab", "laser_name", "n_registros", "n_mids",
                    "score_promedio", "score_std", "snr_promedio",
                    "cv_promedio", "pct_fuera_rango_promedio"
                ]
            )        

        # ---- Write
        if write:
            df_resultados.to_csv(resultados_csv, index=False, encoding="utf-8")
            df_full.to_csv(completos_csv, index=False, encoding="utf-8")
            df_scores.to_csv(scores_csv, index=False, encoding="utf-8")
            df_summary_by_lab.to_csv(summary_by_lab_csv, index=False, encoding="utf-8")
            df_summary_by_date_lab.to_csv(summary_by_date_lab_csv, index=False, encoding="utf-8")
            df_summary_by_laser.to_csv(summary_by_laser_csv, index=False, encoding="utf-8")
            self._write_summary(
                resumen_txt,
                df_scores=df_scores,
                df_skipped=df_skipped,
                df_summary_by_lab=df_summary_by_lab,
                threshold=pass_threshold,
            )
        # ---- Minimal console output
        if verbose:
            n_files = int(df_scores.shape[0])
            n_pass = int(df_scores["pass_to_level2"].sum()) if n_files else 0
            n_skip = int(df_skipped.shape[0])
            print(
                f"[QualityRunner] Files scored: {n_files} | PASS >= {pass_threshold:.2f}: {n_pass} | skipped: {n_skip}"
            )
            #

            print(f"[QualityRunner] resultados_luces.csv: {resultados_csv}")
            print(f"[QualityRunner] quality_scores_by_file.csv: {scores_csv}")
            print(f"[QualityRunner] quality_summary_by_lab.csv: {summary_by_lab_csv}")
            print(f"[QualityRunner] quality_summary_by_date_lab.csv: {summary_by_date_lab_csv}")
            print(f"[QualityRunner] quality_summary_by_laser.csv: {summary_by_laser_csv}")
            print(f"[QualityRunner] resumen_ejecutivo.txt: {resumen_txt}")
        #
                   
        return QualityRunArtifacts(
            output_dir=out_dir,
            resultados_luces_csv=resultados_csv,
            datos_completos_luces_csv=completos_csv,
            scores_by_file_csv=scores_csv,
            summary_by_lab_csv=summary_by_lab_csv,
            summary_by_date_lab_csv=summary_by_date_lab_csv,
            summary_by_laser_csv=summary_by_laser_csv,
            resumen_ejecutivo_txt=resumen_txt,
        )
        
    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    @staticmethod
    def _safe_str(x: Any, default: str = "UNK") -> str:
        if x is None:
            return default
        s = str(x)
        return s if s else default

    @staticmethod
    def _safe_int(x: Any, default: int = -1) -> int:
        try:
            if x is None:
                return default
            return int(x)
        except Exception:
            return default

    def _extract_meta(self, row: Any) -> Dict[str, Any]:
        """
        Extract metadata from a namedtuple row (from itertuples).

        We keep optional fields when present to preserve compatibility,
        but we only require (mid, fecha, lab).
        """
        mid = self._safe_str(getattr(row, "mid", None))
        fecha = self._safe_str(getattr(row, "fecha", None))
        lab = self._safe_str(getattr(row, "lab", None))
        jornada = self._safe_str(getattr(row, "jornada", None), default="UNK")

        # Optional (may exist depending on upstream modes)
        turno = self._safe_str(getattr(row, "turno", None), default="UNK")
        etiqueta = self._safe_str(getattr(row, "etiqueta", None), default="UNK")
        archivo = self._safe_int(getattr(row, "archivo", None), default=-1)
        color = self._safe_str(getattr(row, "color", None), default="unk").lower()

        meta: Dict[str, Any] = {
            "mid": mid,
            "fecha": fecha,
            "lab": lab,
            "turno": turno,
            "jornada": jornada,
            "etiqueta": etiqueta,
            "archivo": archivo,
            "color": color,
        }

        # Preserve any additional columns that might be useful downstream
        # without forcing a contract change.
        for name in ("grupo", "group", "label", "modo", "mode"):
            if hasattr(row, name) and name not in meta:
                meta[name] = getattr(row, name)

        return meta

    def _resolve_parquet_path(self, meta: Dict[str, Any]) -> Path:
        """
        Resolve the Parquet path for a catalog row.

        Preferred (v3 current) naming strategy:
        <analysis_root>/<fecha>/<lab>/<mid>.parquet

        Legacy fallback:
        <analysis_root>/<fecha>/<lab>/<jornada>/<archivo:02d>med<color>.parquet
        """
        base_v3 = self.analysis_root / meta["fecha"] / meta["lab"]

        mid = str(meta["mid"])
        preferred = base_v3 / f"{mid}.parquet"
        if preferred.exists():
            return preferred

        # Legacy fallback (if we have jornada + archivo/color)
        jornada = str(meta.get("jornada", "UNK"))
        base_legacy = self.analysis_root / meta["fecha"] / meta["lab"] / jornada

        archivo = meta.get("archivo", -1)
        color = str(meta.get("color", "unk")).lower()
        if isinstance(archivo, int) and archivo >= 0 and color and color != "unk":
            cands = [
                base_legacy / f"{archivo:02d}med{color}.parquet",
                base_legacy / f"{archivo:03d}med{color}.parquet",
            ]
            for p in cands:
                if p.exists():
                    return p

        return preferred

    def _desired_channels(self) -> Optional[Tuple[str, ...]]:
        ch = getattr(self.cfg.channels, "channels", None)
        if ch is None:
            return None
        return tuple(ch)

    def _detect_channels(
        self,
        *,
        parquet_path: Path,
        desired_channels: Optional[Tuple[str, ...]],
        allowed_prefixes: Tuple[str, ...],
    ) -> List[str]:
        """
        Determine channel columns to read.

        If desired_channels is provided, we intersect with actual schema columns.
        Otherwise, we auto-detect based on allowed_prefixes.
        """
        cols = self._parquet_columns(parquet_path)

        if desired_channels is not None:
            return [c for c in desired_channels if c in cols]

        # Auto-detect
        prefixes = tuple(str(p) for p in allowed_prefixes) if allowed_prefixes else ("Luz",)
        return [c for c in cols if str(c).startswith(prefixes)]

    def _parquet_columns(self, parquet_path: Path) -> List[str]:
        """
        Return Parquet column names with minimal IO.

        Uses pyarrow metadata when available; otherwise falls back to loading a small
        subset (or full table as last resort).
        """
        if pq is not None:
            try:
                pf = pq.ParquetFile(parquet_path)
                return list(pf.schema.names)
            except Exception:
                pass

        # Fallback: read with pandas just to discover columns (may be heavier).
        try:
            df = pd.read_parquet(parquet_path)
            cols = list(df.columns)
            del df
            return cols
        except Exception:
            return []

    def _clean_series(self, df: pd.DataFrame, col: str) -> Optional[pd.Series]:
        if col not in df.columns:
            return None

        s = pd.to_numeric(df[col], errors="coerce")

        # Sentinels -> NA
        sentinels = tuple(getattr(self.cfg.cleaning, "sentinels", (-111.0,)))
        if sentinels:
            s = s.mask(s.isin(sentinels), pd.NA)

        # Drop invalid
        return s.dropna()

    def _load_schema_ok_map(self, schema_by_file_csv: Optional[Path]) -> Dict[str, bool]:
        """
        Load a schema_ok mapping.

        Supports:
          - current schema table: columns ["mid", "schema_ok"]
          - legacy schema table: columns ["parquet_path", "schema_ok"] (ignored here)
        """
        if schema_by_file_csv is None:
            return {}

        p = Path(schema_by_file_csv)
        if not p.exists():
            return {}

        try:
            s = pd.read_csv(p)
        except Exception:
            return {}

        if "mid" in s.columns and "schema_ok" in s.columns:
            out: Dict[str, bool] = {}
            for mid, ok in zip(s["mid"].astype(str), s["schema_ok"].astype(bool)):
                out[str(mid)] = bool(ok)
            return out

        # If the schema file is legacy path-based, we can't reliably map without
        # reconstructing exact paths, so we skip gating in that case.
        return {}

    @staticmethod
    def _row_base(meta: Dict[str, Any], parquet_path: str, columna: str, *, status: str) -> Dict[str, Any]:
        return {
            **meta,
            "parquet_path": parquet_path,
            "laser_name": columna,
            "columna": columna,
            "channel_detected": False,
            "status": status,
        }    

    def _write_summary(
        self,
        path: Path,
        df_scores: pd.DataFrame,
        df_skipped: pd.DataFrame,
        df_summary_by_lab: pd.DataFrame,
        threshold: float,
    ) -> None:
        
        n_files = int(df_scores.shape[0])
        n_pass = int(df_scores["pass_to_level2"].sum()) if n_files and "pass_to_level2" in df_scores.columns else 0
        n_fail = n_files - n_pass
        n_skip = int(df_skipped.shape[0])

        # Worst scores (up to 10)
        worst_lines: List[str] = []
        if n_files and "score_medicion" in df_scores.columns:
            worst_df = df_scores.sort_values("score_medicion", ascending=True).head(10)
            for r in worst_df.itertuples(index=False):
                mid = getattr(r, "mid", "UNK")
                fecha = getattr(r, "fecha", "UNK")
                lab = getattr(r, "lab", "UNK")
                score = getattr(r, "score_medicion", 0.0)
                worst_lines.append(f"- {mid} | {fecha} {lab} | score={float(score):.3f}")
                
        # Skips breakdown
        breakdown_lines: List[str] = []
        if n_skip and "skip_reason" in df_skipped.columns:
            vc = df_skipped["skip_reason"].value_counts()
            for k, v in vc.items():
                breakdown_lines.append(f"- {str(k)}: {int(v)}")

        lab_lines: List[str] = []
        if not df_summary_by_lab.empty and "score_promedio" in df_summary_by_lab.columns:
            top_lab = df_summary_by_lab.sort_values("score_promedio", ascending=False)
            for r in top_lab.itertuples(index=False):
                lab = getattr(r, "lab", "UNK")
                score = getattr(r, "score_promedio", 0.0)
                n_mids = getattr(r, "n_mids", 0)
                lab_lines.append(f"- {lab} | score_promedio={float(score):.3f} | n_mids={int(n_mids)}")

        text = (
            "=== Level 1A: Executive Summary (Quality) ===\n"
            f"PASS threshold (pass_to_level2): {threshold:.2f}\n\n"
            f"Files scored: {n_files}\n"
            f"PASS: {n_pass}\n"
            f"FAIL: {n_fail}\n"
            f"Skipped: {n_skip}\n\n"
            "Skipped breakdown:\n"
            f"{chr(10).join(breakdown_lines) if breakdown_lines else '- (none)'}\n\n"
            "Worst scores (up to 10):\n"
            f"{chr(10).join(worst_lines) if worst_lines else '- (none)'}\n\n"
            "Lab summary:\n"
            f"{chr(10).join(lab_lines) if lab_lines else '- (none)'}\n"
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")