from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityCompareConfig:
    """
    Configuración del bloque visual/comparativo de Quality.

    max_points:
        Máximo de puntos por curva al graficar series temporales.
    dpi:
        Resolución de salida para PNG.
    output_dir_name:
        Subcarpeta por defecto dentro de Reports/Level1_Quality.
    """
    max_points: int = 3000
    dpi: int = 150
    output_dir_name: str = "Compare"


@dataclass(frozen=True)
class QualityCompareArtifacts:
    output_dir: Path
    global_dir: Path
    per_measurement_dir: Path
    per_laser_dir: Path
    per_lab_dir: Path


class QualityCompareRunner:
    """
    Generate comparative visual reports from Quality outputs.

    Inputs:
      - resultados_luces.csv
      - analysis_ready_schema_by_file.csv (optional)
      - Analysis Ready parquets

    Outputs:
      - comparacion_global/
      - Per_Measurement/<fecha>/<lab>/<mid>/
      - Per_Laser/<laser_name>/
      - Per_Lab/ (reserved for future extension)

    Notes:
      - This module does NOT recompute metrics.
      - It is designed for multi-lab visual traceability.
      - It generates compact global summaries and per-measurement time-series reports.
    """

    def __init__(self, config: Optional[QualityCompareConfig] = None) -> None:
        self.cfg = config or QualityCompareConfig()

    @staticmethod
    def _sanitize_stem(s: str) -> str:
        s = str(s).strip()
        if not s:
            return "UNK"
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
        return s[:64]

    def _measurement_dir(self, output_dir: Path, fecha: str, lab: str, mid: str) -> Path:
        return output_dir / "Per_Measurement" / str(fecha) / str(lab) / str(mid)

    def _laser_dir(self, output_dir: Path, laser_name: str) -> Path:
        return output_dir / "Per_Laser" / self._sanitize_stem(str(laser_name))

    def _lab_dir(self, output_dir: Path, lab: str) -> Path:
        return output_dir / "Per_Lab" / self._sanitize_stem(str(lab))

    @staticmethod
    def _pick_time_column(df: pd.DataFrame) -> Optional[str]:
        for c in ["t_rel", "t_sys", "Time", "Tiempo", "time", "time_rel", "time_sys"]:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _normalize_series(y: pd.Series) -> pd.Series:
        s = pd.to_numeric(y, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return s
        ymin = float(s.min())
        ymax = float(s.max())
        if ymax <= ymin:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - ymin) / (ymax - ymin)

    @staticmethod
    def _downsample_xy(x: pd.Series, y: pd.Series, max_points: int) -> Tuple[pd.Series, pd.Series]:
        n = min(len(x), len(y))
        if n <= max_points or max_points <= 0:
            return x.iloc[:n], y.iloc[:n]
        idx = np.linspace(0, n - 1, num=max_points, dtype=int)
        return x.iloc[idx], y.iloc[idx]

    @staticmethod
    def _normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        rename_map: Dict[str, str] = {}

        if "columna" in out.columns and "laser_name" not in out.columns:
            rename_map["columna"] = "laser_name"

        if "calidad_general" in out.columns and "score_canal" not in out.columns:
            rename_map["calidad_general"] = "score_canal"

        if rename_map:
            out = out.rename(columns=rename_map)

        for c in ["mid", "fecha", "lab", "laser_name", "parquet_path"]:
            if c in out.columns:
                out[c] = out[c].astype(str).str.strip()

        return out

    @staticmethod
    def _normalize_schema_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for c in ["mid", "fecha", "lab", "parquet_path"]:
            if c in out.columns:
                out[c] = out[c].astype(str).str.strip()

        return out

    @staticmethod
    def _merge_meta(df_results: pd.DataFrame, df_schema: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df_schema is None or df_schema.empty:
            return df_results

        res = df_results.copy()
        sch = df_schema.copy()

        # Primer intento: merge por mid
        if "mid" in res.columns and "mid" in sch.columns:
            extra_cols = [c for c in sch.columns if c not in res.columns or c == "mid"]
            sch_mid = sch[extra_cols].drop_duplicates(subset=["mid"])
            res = res.merge(sch_mid, on="mid", how="left", suffixes=("", "_schema"))

        # Segundo intento: rellenar por parquet_path si sigue faltando metadata
        if "parquet_path" in res.columns and "parquet_path" in sch.columns:
            extra_cols = [c for c in sch.columns if c not in res.columns or c == "parquet_path"]
            sch_pq = sch[extra_cols].drop_duplicates(subset=["parquet_path"])
            res = res.merge(sch_pq, on="parquet_path", how="left", suffixes=("", "_schema2"))

        return res

    def _plot_heatmap_mid_laser(
        self,
        df: pd.DataFrame,
        *,
        metric: str,
        out_path: Path,
    ) -> None:
        req = {"mid", "laser_name", metric}
        if not req.issubset(df.columns):
            return

        tmp = df.copy()
        tmp = tmp.dropna(subset=["mid", "laser_name", metric])
        if tmp.empty:
            return

        piv = tmp.pivot_table(
            index="mid",
            columns="laser_name",
            values=metric,
            aggfunc="mean",
        )

        if piv.empty:
            return

        plt.figure(figsize=(12, max(6, 0.22 * len(piv))))
        plt.imshow(piv.values, aspect="auto")
        plt.colorbar(label=metric)
        plt.xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
        plt.yticks(range(len(piv.index)), piv.index)
        plt.title(f"Heatmap {metric}: mid × laser")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _plot_quality_by_lab(
        self,
        df: pd.DataFrame,
        *,
        metric: str,
        out_path: Path,
    ) -> None:
        req = {"lab", metric}
        if not req.issubset(df.columns):
            return

        tmp = df.copy().dropna(subset=["lab", metric])
        if tmp.empty:
            return

        order = sorted(tmp["lab"].astype(str).unique())
        data = [pd.to_numeric(tmp.loc[tmp["lab"] == lab, metric], errors="coerce").dropna() for lab in order]
        if not any(len(x) > 0 for x in data):
            return

        plt.figure(figsize=(max(8, 1.4 * len(order)), 6))
        plt.boxplot(data, labels=order)
        plt.title(f"{metric} by lab")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _plot_metric_by_laser(
        self,
        df: pd.DataFrame,
        *,
        metric: str,
        out_path: Path,
    ) -> None:
        req = {"laser_name", metric}
        if not req.issubset(df.columns):
            return

        tmp = df.copy().dropna(subset=["laser_name", metric])
        if tmp.empty:
            return

        order = sorted(tmp["laser_name"].astype(str).unique())
        data = [pd.to_numeric(tmp.loc[tmp["laser_name"] == k, metric], errors="coerce").dropna() for k in order]
        if not any(len(x) > 0 for x in data):
            return

        plt.figure(figsize=(max(8, 1.4 * len(order)), 6))
        plt.boxplot(data, labels=order)
        plt.title(f"{metric} by laser")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _plot_heatmap_date_lab_laser(
        self,
        df: pd.DataFrame,
        *,
        metric: str,
        out_path: Path,
    ) -> None:
        req = {"fecha", "lab", "laser_name", metric}
        if not req.issubset(df.columns):
            return

        tmp = df.copy().dropna(subset=["fecha", "lab", "laser_name", metric])
        if tmp.empty:
            return

        tmp["date_lab"] = tmp["fecha"].astype(str) + " | " + tmp["lab"].astype(str)
        piv = tmp.pivot_table(
            index="date_lab",
            columns="laser_name",
            values=metric,
            aggfunc="mean",
        )

        if piv.empty:
            return

        plt.figure(figsize=(12, max(6, 0.35 * len(piv))))
        plt.imshow(piv.values, aspect="auto")
        plt.colorbar(label=metric)
        plt.xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
        plt.yticks(range(len(piv.index)), piv.index)
        plt.title(f"Heatmap {metric}: (fecha, lab) × laser")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _generate_global_compact_reports(
        self,
        df_results: pd.DataFrame,
        *,
        output_dir: Path,
    ) -> Path:
        global_dir = output_dir / "comparacion_global"
        global_dir.mkdir(parents=True, exist_ok=True)

        metric_score = "score_canal" if "score_canal" in df_results.columns else "calidad_general"

        self._plot_heatmap_mid_laser(
            df_results,
            metric=metric_score,
            out_path=global_dir / "heatmap_mid_laser_score.png",
        )

        self._plot_quality_by_lab(
            df_results,
            metric=metric_score,
            out_path=global_dir / "boxplot_score_by_lab.png",
        )

        if "snr_db" in df_results.columns:
            self._plot_metric_by_laser(
                df_results,
                metric="snr_db",
                out_path=global_dir / "boxplot_snr_by_laser.png",
            )

        self._plot_heatmap_date_lab_laser(
            df_results,
            metric=metric_score,
            out_path=global_dir / "heatmap_date_lab_laser.png",
        )

        return global_dir

    def _load_measurement_table(self, parquet_path: Path) -> Optional[pd.DataFrame]:
        try:
            if parquet_path.exists():
                return pd.read_parquet(parquet_path)
        except Exception:
            return None
        return None

    def _plot_measurement_individual_series(
        self,
        df_measure: pd.DataFrame,
        *,
        time_col: str,
        channels: List[str],
        out_path: Path,
    ) -> None:
        if not channels:
            return

        n = len(channels)
        fig, axes = plt.subplots(n, 1, figsize=(12, max(2.6 * n, 4)), squeeze=False)
        axes = axes.flatten()

        for ax, ch in zip(axes, channels):
            x = pd.to_numeric(df_measure[time_col], errors="coerce")
            y = pd.to_numeric(df_measure[ch], errors="coerce")
            mask = (~x.isna()) & (~y.isna())
            x = x[mask]
            y = y[mask]
            if x.empty or y.empty:
                ax.set_title(f"{ch} (sin datos válidos)")
                continue
            x, y = self._downsample_xy(x, y, self.cfg.max_points)
            ax.plot(x, y)
            ax.set_title(ch)
            ax.set_xlabel(time_col)
            ax.set_ylabel("signal")

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _plot_measurement_overlay_raw(
        self,
        df_measure: pd.DataFrame,
        *,
        time_col: str,
        channels: List[str],
        out_path: Path,
    ) -> None:
        if not channels:
            return

        plt.figure(figsize=(12, 6))
        for ch in channels:
            x = pd.to_numeric(df_measure[time_col], errors="coerce")
            y = pd.to_numeric(df_measure[ch], errors="coerce")
            mask = (~x.isna()) & (~y.isna())
            x = x[mask]
            y = y[mask]
            if x.empty or y.empty:
                continue
            x, y = self._downsample_xy(x, y, self.cfg.max_points)
            plt.plot(x, y, label=ch)

        plt.title("Serie encimada por medición (escala original)")
        plt.xlabel(time_col)
        plt.ylabel("signal")
        plt.legend()
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _plot_measurement_overlay_normalized(
        self,
        df_measure: pd.DataFrame,
        *,
        time_col: str,
        channels: List[str],
        out_path: Path,
    ) -> None:
        if not channels:
            return

        plt.figure(figsize=(12, 6))
        for ch in channels:
            x = pd.to_numeric(df_measure[time_col], errors="coerce")
            y = pd.to_numeric(df_measure[ch], errors="coerce")
            mask = (~x.isna()) & (~y.isna())
            x = x[mask]
            y = y[mask]
            if x.empty or y.empty:
                continue
            y = self._normalize_series(y)
            if y.empty:
                continue
            x = x.loc[y.index]
            x, y = self._downsample_xy(x, y, self.cfg.max_points)
            plt.plot(x, y, label=ch)

        plt.title("Serie encimada por medición (normalizada)")
        plt.xlabel(time_col)
        plt.ylabel("normalized signal")
        plt.legend()
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _write_measurement_summary(
        self,
        df_one_measure: pd.DataFrame,
        *,
        out_path: Path,
    ) -> None:
        if df_one_measure.empty:
            return

        score_col = "score_canal" if "score_canal" in df_one_measure.columns else "calidad_general"
        df_tmp = df_one_measure.copy()

        mid = str(df_tmp["mid"].iloc[0]) if "mid" in df_tmp.columns else "UNK"
        fecha = str(df_tmp["fecha"].iloc[0]) if "fecha" in df_tmp.columns else "UNK"
        lab = str(df_tmp["lab"].iloc[0]) if "lab" in df_tmp.columns else "UNK"

        valid_mask = df_tmp["valido"] == True if "valido" in df_tmp.columns else pd.Series([True] * len(df_tmp), index=df_tmp.index)
        valid = df_tmp.loc[valid_mask].copy()
        n_valid = len(valid)

        score_med = float(pd.to_numeric(valid[score_col], errors="coerce").mean()) if n_valid > 0 else float("nan")

        best_line = "- (none)"
        worst_line = "- (none)"
        if n_valid > 0 and score_col in valid.columns and "laser_name" in valid.columns:
            valid2 = valid.dropna(subset=[score_col, "laser_name"]).copy()
            if not valid2.empty:
                best = valid2.sort_values(score_col, ascending=False).iloc[0]
                worst = valid2.sort_values(score_col, ascending=True).iloc[0]
                best_line = f"{best['laser_name']} | {score_col}={float(best[score_col]):.3f}"
                worst_line = f"{worst['laser_name']} | {score_col}={float(worst[score_col]):.3f}"

        txt = (
            f"mid: {mid}\n"
            f"fecha: {fecha}\n"
            f"lab: {lab}\n"
            f"score_medicion: {score_med:.3f}\n"
            f"n_canales_validos: {n_valid}\n"
            f"mejor_laser: {best_line}\n"
            f"peor_laser: {worst_line}\n"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(txt, encoding="utf-8")

    def _generate_per_measurement_reports(
        self,
        df_results: pd.DataFrame,
        *,
        output_dir: Path,
    ) -> Path:
        base_dir = output_dir / "Per_Measurement"
        base_dir.mkdir(parents=True, exist_ok=True)

        if df_results.empty or "parquet_path" not in df_results.columns or "mid" not in df_results.columns:
            return base_dir

        for mid, g in df_results.groupby("mid", dropna=False):
            one = g.copy()
            parquet_path = Path(str(one["parquet_path"].dropna().iloc[0])) if one["parquet_path"].notna().any() else None
            if parquet_path is None:
                continue

            df_measure = self._load_measurement_table(parquet_path)
            if df_measure is None or df_measure.empty:
                continue

            time_col = self._pick_time_column(df_measure)
            if time_col is None:
                continue

            if "laser_name" in one.columns:
                channels = [c for c in one["laser_name"].dropna().astype(str).unique() if c in df_measure.columns]
            else:
                channels = []

            if not channels:
                channels = [c for c in df_measure.columns if str(c).startswith(("Laser", "Luz", "Ch", "Canal"))]

            fecha = str(one["fecha"].iloc[0]) if "fecha" in one.columns else "UNK"
            lab = str(one["lab"].iloc[0]) if "lab" in one.columns else "UNK"
            mdir = self._measurement_dir(output_dir, fecha, lab, str(mid))
            mdir.mkdir(parents=True, exist_ok=True)

            self._plot_measurement_individual_series(
                df_measure,
                time_col=time_col,
                channels=list(channels),
                out_path=mdir / "series_individuales.png",
            )

            self._plot_measurement_overlay_raw(
                df_measure,
                time_col=time_col,
                channels=list(channels),
                out_path=mdir / "serie_encimada_original.png",
            )

            self._plot_measurement_overlay_normalized(
                df_measure,
                time_col=time_col,
                channels=list(channels),
                out_path=mdir / "serie_encimada_normalizada.png",
            )

            self._write_measurement_summary(
                one,
                out_path=mdir / "resumen_medicion.txt",
            )

        return base_dir

    def _plot_laser_score_by_lab(
        self,
        df_laser: pd.DataFrame,
        *,
        out_path: Path,
    ) -> None:
        req = {"lab", "score_canal"}
        if not req.issubset(df_laser.columns):
            return
        tmp = df_laser.dropna(subset=["lab", "score_canal"]).copy()
        if tmp.empty:
            return

        order = sorted(tmp["lab"].astype(str).unique())
        data = [pd.to_numeric(tmp.loc[tmp["lab"] == lab, "score_canal"], errors="coerce").dropna() for lab in order]
        if not any(len(x) > 0 for x in data):
            return

        plt.figure(figsize=(max(8, 1.4 * len(order)), 6))
        plt.boxplot(data, labels=order)
        plt.title("Score por laboratorio")
        plt.ylabel("score_canal")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _plot_laser_snr_by_lab(
        self,
        df_laser: pd.DataFrame,
        *,
        out_path: Path,
    ) -> None:
        req = {"lab", "snr_db"}
        if not req.issubset(df_laser.columns):
            return
        tmp = df_laser.dropna(subset=["lab", "snr_db"]).copy()
        if tmp.empty:
            return

        order = sorted(tmp["lab"].astype(str).unique())
        data = [pd.to_numeric(tmp.loc[tmp["lab"] == lab, "snr_db"], errors="coerce").dropna() for lab in order]
        if not any(len(x) > 0 for x in data):
            return

        plt.figure(figsize=(max(8, 1.4 * len(order)), 6))
        plt.boxplot(data, labels=order)
        plt.title("SNR por laboratorio")
        plt.ylabel("snr_db")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _plot_laser_score_by_date_lab(
        self,
        df_laser: pd.DataFrame,
        *,
        out_path: Path,
    ) -> None:
        req = {"fecha", "lab", "score_canal"}
        if not req.issubset(df_laser.columns):
            return

        tmp = df_laser.dropna(subset=["fecha", "lab", "score_canal"]).copy()
        if tmp.empty:
            return

        agg = (
            tmp.groupby(["fecha", "lab"], dropna=False)["score_canal"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        for lab, g in agg.groupby("lab", dropna=False):
            g = g.sort_values("fecha")
            plt.plot(g["fecha"].astype(str), g["score_canal"], marker="o", label=str(lab))

        plt.title("Score promedio por fecha y laboratorio")
        plt.xlabel("fecha")
        plt.ylabel("score_canal")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=self.cfg.dpi)
        plt.close()

    def _write_laser_summary(
        self,
        df_laser: pd.DataFrame,
        *,
        out_path: Path,
    ) -> None:
        if df_laser.empty:
            return

        laser_name = str(df_laser["laser_name"].iloc[0]) if "laser_name" in df_laser.columns else "UNK"
        score_mean = float(pd.to_numeric(df_laser.get("score_canal"), errors="coerce").mean()) if "score_canal" in df_laser.columns else float("nan")

        txt = [f"laser_name: {laser_name}", f"score_promedio: {score_mean:.3f}"]

        if {"lab", "score_canal"}.issubset(df_laser.columns):
            agg = (
                df_laser.groupby("lab", dropna=False)["score_canal"]
                .mean()
                .reset_index()
                .sort_values("score_canal", ascending=False)
            )
            txt.append("")
            txt.append("score_promedio_por_lab:")
            for r in agg.itertuples(index=False):
                txt.append(f"- {r.lab}: {float(r.score_canal):.3f}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(txt), encoding="utf-8")

    def _generate_per_laser_reports(
        self,
        df_results: pd.DataFrame,
        *,
        output_dir: Path,
    ) -> Path:
        base_dir = output_dir / "Per_Laser"
        base_dir.mkdir(parents=True, exist_ok=True)

        if df_results.empty or "laser_name" not in df_results.columns:
            return base_dir

        for laser_name, g in df_results.groupby("laser_name", dropna=False):
            if pd.isna(laser_name):
                continue

            ldir = self._laser_dir(output_dir, str(laser_name))
            ldir.mkdir(parents=True, exist_ok=True)

            if "score_canal" in g.columns:
                self._plot_laser_score_by_lab(
                    g,
                    out_path=ldir / "score_por_lab.png",
                )

            if "snr_db" in g.columns:
                self._plot_laser_snr_by_lab(
                    g,
                    out_path=ldir / "snr_por_lab.png",
                )

            if {"fecha", "lab", "score_canal"}.issubset(g.columns):
                self._plot_laser_score_by_date_lab(
                    g,
                    out_path=ldir / "score_por_fecha_lab.png",
                )

            self._write_laser_summary(
                g,
                out_path=ldir / "resumen_laser.txt",
            )

        return base_dir

    def run(
        self,
        *,
        resultados_csv: Path | str,
        schema_by_file_csv: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        verbose: bool = True,
    ) -> QualityCompareArtifacts:
        resultados_csv = Path(resultados_csv)
        if not resultados_csv.exists():
            raise FileNotFoundError(f"resultados_luces.csv not found: {resultados_csv}")

        if output_dir is None:
            output_dir = resultados_csv.parent / self.cfg.output_dir_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df_res = pd.read_csv(resultados_csv)
        df_res = self._normalize_results_df(df_res)

        df_schema: Optional[pd.DataFrame] = None
        if schema_by_file_csv is not None:
            schema_by_file_csv = Path(schema_by_file_csv)
            if schema_by_file_csv.exists():
                df_schema = pd.read_csv(schema_by_file_csv)
                df_schema = self._normalize_schema_df(df_schema)

        df_res = self._merge_meta(df_res, df_schema)

        global_dir = self._generate_global_compact_reports(
            df_res,
            output_dir=output_dir,
        )

        per_measurement_dir = self._generate_per_measurement_reports(
            df_res,
            output_dir=output_dir,
        )

        per_laser_dir = self._generate_per_laser_reports(
            df_res,
            output_dir=output_dir,
        )

        per_lab_dir = output_dir / "Per_Lab"
        per_lab_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"[QualityCompare] global reports:        {global_dir}")
            print(f"[QualityCompare] per-measurement dir:   {per_measurement_dir}")
            print(f"[QualityCompare] per-laser dir:         {per_laser_dir}")
            print(f"[QualityCompare] per-lab dir:           {per_lab_dir}")

        return QualityCompareArtifacts(
            output_dir=output_dir,
            global_dir=global_dir,
            per_measurement_dir=per_measurement_dir,
            per_laser_dir=per_laser_dir,
            per_lab_dir=per_lab_dir,
        )