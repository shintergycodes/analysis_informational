from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from correlation_lasers_config import CorrelationLasersConfig


# ==========================================================
# Correlation Lasers — Compare / Reporting
# ==========================================================
#
# Level 3C — Correlation Lasers Compare
#
# Purpose
# -------
# Read Level 3B tabular artifacts and generate:
# - compact global summaries
# - per-(fecha, lab) heatmaps of laser correlation
# - per-(fecha, lab) average laser profiles
# - per-lab temporal comparisons
# - per-date inter-lab comparisons
# - simple textual summaries
#
# This module does NOT:
# - read raw experiment CSV files
# - depend on Ctrl/Exp semantics
# - use turno/jornada as structural axes
#
# ==========================================================


# ----------------------------------------------------------
# Public artifacts
# ----------------------------------------------------------
@dataclass(frozen=True)
class CorrelationLasersCompareArtifacts:
    output_root: Path
    global_dir: Path
    per_date_lab_dir: Path
    per_laser_dir: Path
    per_lab_dir: Path
    global_summary_csv: Path
    global_summary_txt: Path
    pairwise_dates_summary_csv: Path
    pairwise_labs_summary_csv: Path


# ----------------------------------------------------------
# Compare runner
# ----------------------------------------------------------
class CorrelationLasersCompare:
    def __init__(self, config: CorrelationLasersConfig) -> None:
        self.config = config
        self.config.validate()

    # ======================================================
    # Public API
    # ======================================================
    def run(
        self,
        *,
        summary_by_date_lab_csv: Path,
        correlation_by_date_lab_csv: Path,
        pairwise_dates_by_lab_csv: Path,
        pairwise_labs_by_date_csv: Path,
        profiles_by_measurement_csv: Optional[Path] = None,
        output_dir: Path,
        write: bool = True,
        verbose: bool = False,
    ) -> CorrelationLasersCompareArtifacts:
        output_root = Path(output_dir)
        global_dir = output_root / self.config.artifacts.global_dir
        per_date_lab_dir = output_root / self.config.artifacts.per_date_lab_dir
        per_laser_dir = output_root / self.config.artifacts.per_laser_dir
        per_lab_dir = output_root / self.config.artifacts.per_lab_dir

        if write:
            for d in (output_root, global_dir, per_date_lab_dir, per_laser_dir, per_lab_dir):
                d.mkdir(parents=True, exist_ok=True)

        summary_df = pd.read_csv(summary_by_date_lab_csv)
        corr_df = pd.read_csv(correlation_by_date_lab_csv)
        pair_dates_df = pd.read_csv(pairwise_dates_by_lab_csv)
        pair_labs_df = pd.read_csv(pairwise_labs_by_date_csv)
        profiles_df = (
            pd.read_csv(profiles_by_measurement_csv)
            if profiles_by_measurement_csv is not None and Path(profiles_by_measurement_csv).exists()
            else None
        )

        if verbose:
            print(f"[CORR-CMP] summary_by_date_lab rows : {len(summary_df)}")
            print(f"[CORR-CMP] correlation_by_date_lab : {len(corr_df)}")
            print(f"[CORR-CMP] pairwise_dates_by_lab   : {len(pair_dates_df)}")
            print(f"[CORR-CMP] pairwise_labs_by_date   : {len(pair_labs_df)}")
            if profiles_df is not None:
                print(f"[CORR-CMP] profiles_by_measurement : {len(profiles_df)}")

        global_summary_df = self._build_global_summary(summary_df, corr_df, pair_dates_df, pair_labs_df)
        pairwise_dates_summary_df = self._build_pairwise_dates_summary(pair_dates_df)
        pairwise_labs_summary_df = self._build_pairwise_labs_summary(pair_labs_df)

        global_summary_csv = global_dir / "global_summary.csv"
        global_summary_txt = global_dir / "resumen_global.txt"
        pairwise_dates_summary_csv = global_dir / "pairwise_dates_summary.csv"
        pairwise_labs_summary_csv = global_dir / "pairwise_labs_summary.csv"

        if write:
            global_summary_df.to_csv(global_summary_csv, index=False, encoding="utf-8-sig")
            pairwise_dates_summary_df.to_csv(pairwise_dates_summary_csv, index=False, encoding="utf-8-sig")
            pairwise_labs_summary_df.to_csv(pairwise_labs_summary_csv, index=False, encoding="utf-8-sig")
            global_summary_txt.write_text(
                self._render_global_summary_txt(global_summary_df, pairwise_dates_summary_df, pairwise_labs_summary_df),
                encoding="utf-8",
            )

        self._build_per_date_lab_reports(summary_df, corr_df, per_date_lab_dir, write=write, verbose=verbose)
        self._build_per_lab_reports(pair_dates_df, per_lab_dir, write=write, verbose=verbose)
        self._build_per_laser_reports(summary_df, pair_dates_df, pair_labs_df, per_laser_dir, write=write, verbose=verbose)
        self._build_global_figures(global_summary_df, pairwise_dates_summary_df, pairwise_labs_summary_df, global_dir, write=write)

        return CorrelationLasersCompareArtifacts(
            output_root=output_root,
            global_dir=global_dir,
            per_date_lab_dir=per_date_lab_dir,
            per_laser_dir=per_laser_dir,
            per_lab_dir=per_lab_dir,
            global_summary_csv=global_summary_csv,
            global_summary_txt=global_summary_txt,
            pairwise_dates_summary_csv=pairwise_dates_summary_csv,
            pairwise_labs_summary_csv=pairwise_labs_summary_csv,
        )

    # ======================================================
    # Global summaries
    # ======================================================
    def _build_global_summary(
        self,
        summary_df: pd.DataFrame,
        corr_df: pd.DataFrame,
        pair_dates_df: pd.DataFrame,
        pair_labs_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []

        metric_cols = list(self.config.metrics)

        for _, row in summary_df.iterrows():
            out: Dict[str, object] = {
                "fecha": row["fecha"],
                "lab": row["lab"],
                "n_measurements": row.get("n_measurements", np.nan),
            }

            for metric in metric_cols:
                out[f"{metric}__profile_mean"] = row.get(f"{metric}__profile_mean", np.nan)
                out[f"{metric}__profile_std"] = row.get(f"{metric}__profile_std", np.nan)
                out[f"{metric}__anisotropy_ratio"] = row.get(f"{metric}__anisotropy_ratio", np.nan)

                mask = (
                    (corr_df["fecha"] == row["fecha"])
                    & (corr_df["lab"] == row["lab"])
                    & (corr_df["metric"] == metric)
                )
                sub = corr_df.loc[mask].copy()
                out[f"{metric}__mean_pair_corr"] = (
                    float(sub["corr_value"].mean()) if not sub.empty else np.nan
                )

            rows.append(out)

        out_df = pd.DataFrame(rows)
        if out_df.empty:
            return out_df

        out_df["global_mean_anisotropy"] = out_df[
            [f"{m}__anisotropy_ratio" for m in metric_cols if f"{m}__anisotropy_ratio" in out_df.columns]
        ].mean(axis=1)

        out_df["global_mean_pair_corr"] = out_df[
            [f"{m}__mean_pair_corr" for m in metric_cols if f"{m}__mean_pair_corr" in out_df.columns]
        ].mean(axis=1)

        out_df["global_profile_std_mean"] = out_df[
            [f"{m}__profile_std" for m in metric_cols if f"{m}__profile_std" in out_df.columns]
        ].mean(axis=1)

        out_df = out_df.sort_values(["fecha", "lab"]).reset_index(drop=True)
        return out_df

    def _build_pairwise_dates_summary(self, pair_dates_df: pd.DataFrame) -> pd.DataFrame:
        if pair_dates_df.empty:
            return pair_dates_df

        df = pair_dates_df.copy()
        corr_cols = [c for c in df.columns if c.endswith("__profile_corr")]
        dist_cols = [c for c in df.columns if c.endswith("__profile_distance")]

        if corr_cols:
            df["pairwise_profile_corr_mean_recomputed"] = df[corr_cols].mean(axis=1)
        if dist_cols:
            df["pairwise_profile_distance_mean_recomputed"] = df[dist_cols].mean(axis=1)

        return df.sort_values(["lab", "fecha_a", "fecha_b"]).reset_index(drop=True)

    def _build_pairwise_labs_summary(self, pair_labs_df: pd.DataFrame) -> pd.DataFrame:
        if pair_labs_df.empty:
            return pair_labs_df

        df = pair_labs_df.copy()
        corr_cols = [c for c in df.columns if c.endswith("__profile_corr")]
        dist_cols = [c for c in df.columns if c.endswith("__profile_distance")]

        if corr_cols:
            df["pairwise_profile_corr_mean_recomputed"] = df[corr_cols].mean(axis=1)
        if dist_cols:
            df["pairwise_profile_distance_mean_recomputed"] = df[dist_cols].mean(axis=1)

        return df.sort_values(["fecha", "lab_a", "lab_b"]).reset_index(drop=True)

    def _render_global_summary_txt(
        self,
        global_summary_df: pd.DataFrame,
        pairwise_dates_summary_df: pd.DataFrame,
        pairwise_labs_summary_df: pd.DataFrame,
    ) -> str:
        lines: List[str] = []
        lines.append("Correlation Lasers — Resumen global")
        lines.append("=" * 48)
        lines.append("")
        lines.append(f"Bloques (fecha, lab): {len(global_summary_df)}")
        lines.append(f"Comparaciones fecha-por-lab: {len(pairwise_dates_summary_df)}")
        lines.append(f"Comparaciones lab-por-fecha: {len(pairwise_labs_summary_df)}")
        lines.append("")

        if not global_summary_df.empty:
            most_aniso = global_summary_df.sort_values("global_mean_anisotropy", ascending=False).iloc[0]
            best_corr = global_summary_df.sort_values("global_mean_pair_corr", ascending=False).iloc[0]

            lines.append("Bloque con mayor anisotropía media:")
            lines.append(
                f"  - {most_aniso['fecha']} | {most_aniso['lab']} | "
                f"anisotropía media={_fmt(most_aniso.get('global_mean_anisotropy'))}"
            )
            lines.append("")
            lines.append("Bloque con mayor correlación media entre pares de láseres:")
            lines.append(
                f"  - {best_corr['fecha']} | {best_corr['lab']} | "
                f"corr media={_fmt(best_corr.get('global_mean_pair_corr'))}"
            )
            lines.append("")

        if not pairwise_dates_summary_df.empty:
            best_dates = pairwise_dates_summary_df.sort_values(
                "pairwise_profile_corr_mean", ascending=False
            ).iloc[0]
            lines.append("Par temporal más parecido dentro de laboratorio:")
            lines.append(
                f"  - {best_dates['lab']} | {best_dates['fecha_a']} vs {best_dates['fecha_b']} | "
                f"corr media={_fmt(best_dates.get('pairwise_profile_corr_mean'))}"
            )
            lines.append("")

        if not pairwise_labs_summary_df.empty:
            best_labs = pairwise_labs_summary_df.sort_values(
                "pairwise_profile_corr_mean", ascending=False
            ).iloc[0]
            lines.append("Par de laboratorios más parecido dentro de una fecha:")
            lines.append(
                f"  - {best_labs['fecha']} | {best_labs['lab_a']} vs {best_labs['lab_b']} | "
                f"corr media={_fmt(best_labs.get('pairwise_profile_corr_mean'))}"
            )

        return "\n".join(lines)

    # ======================================================
    # Per (fecha, lab)
    # ======================================================
    def _build_per_date_lab_reports(
        self,
        summary_df: pd.DataFrame,
        corr_df: pd.DataFrame,
        out_dir: Path,
        *,
        write: bool,
        verbose: bool,
    ) -> None:
        if summary_df.empty:
            return

        for _, row in summary_df.iterrows():
            fecha = str(row["fecha"])
            lab = str(row["lab"])
            block_dir = out_dir / f"{fecha}_{lab}"
            if write:
                block_dir.mkdir(parents=True, exist_ok=True)

            summary_txt = block_dir / "resumen_bloque.txt"
            self._write_block_summary_txt(row, summary_txt, write=write)

            for metric in self.config.metrics:
                heatmap_path = block_dir / f"heatmap_corr_{metric}.png"
                profile_path = block_dir / f"profile_{metric}.png"

                sub = corr_df[
                    (corr_df["fecha"] == fecha)
                    & (corr_df["lab"] == lab)
                    & (corr_df["metric"] == metric)
                ].copy()

                mat = self._pairwise_corr_to_matrix(sub, expected_lasers_count=self.config.expected_lasers_count)
                self._plot_heatmap(
                    matrix=mat,
                    labels=[f"Laser_{i}" for i in range(1, self.config.expected_lasers_count + 1)],
                    title=f"{fecha} | {lab} | {metric} — correlación entre láseres",
                    save_path=heatmap_path,
                    write=write,
                )

                vec = _summary_profile_vector(row, metric, self.config.expected_lasers_count)
                self._plot_profile(
                    values=vec,
                    title=f"{fecha} | {lab} | {metric} — perfil medio",
                    ylabel=metric,
                    save_path=profile_path,
                    write=write,
                )

                if verbose:
                    print(f"[CORR-CMP][OK] {block_dir.name} | {metric}")

    def _write_block_summary_txt(self, row: pd.Series, path: Path, *, write: bool) -> None:
        lines = []
        lines.append("Resumen de bloque Correlation Lasers")
        lines.append("=" * 40)
        lines.append(f"fecha: {row['fecha']}")
        lines.append(f"lab  : {row['lab']}")
        lines.append(f"n_measurements: {row.get('n_measurements', 'NA')}")
        lines.append("")

        for metric in self.config.metrics:
            lines.append(f"[{metric}]")
            lines.append(f"  profile_mean      : {_fmt(row.get(f'{metric}__profile_mean'))}")
            lines.append(f"  profile_std       : {_fmt(row.get(f'{metric}__profile_std'))}")
            lines.append(f"  anisotropy_ratio  : {_fmt(row.get(f'{metric}__anisotropy_ratio'))}")
            lines.append("")

        if write:
            path.write_text("\n".join(lines), encoding="utf-8")

    # ======================================================
    # Per lab
    # ======================================================
    def _build_per_lab_reports(
        self,
        pair_dates_df: pd.DataFrame,
        out_dir: Path,
        *,
        write: bool,
        verbose: bool,
    ) -> None:
        if pair_dates_df.empty:
            return

        for lab, g in pair_dates_df.groupby("lab", dropna=False):
            lab_dir = out_dir / str(lab)
            if write:
                lab_dir.mkdir(parents=True, exist_ok=True)

            for metric in self.config.metrics:
                y_corr = g.get(f"{metric}__profile_corr")
                if y_corr is None:
                    continue

                labels = [f"{a} vs {b}" for a, b in zip(g["fecha_a"], g["fecha_b"])]
                self._plot_bar(
                    labels=labels,
                    values=g[f"{metric}__profile_corr"].to_numpy(dtype=float),
                    title=f"{lab} | {metric} — correlación entre fechas",
                    ylabel="corr",
                    save_path=lab_dir / f"{metric}_corr_temporal.png",
                    write=write,
                )
                self._plot_bar(
                    labels=labels,
                    values=g[f"{metric}__profile_distance"].to_numpy(dtype=float),
                    title=f"{lab} | {metric} — distancia entre fechas",
                    ylabel="distance",
                    save_path=lab_dir / f"{metric}_distance_temporal.png",
                    write=write,
                )

            if verbose:
                print(f"[CORR-CMP][OK] Per_Lab/{lab}")

    # ======================================================
    # Per laser
    # ======================================================
    def _build_per_laser_reports(
        self,
        summary_df: pd.DataFrame,
        pair_dates_df: pd.DataFrame,
        pair_labs_df: pd.DataFrame,
        out_dir: Path,
        *,
        write: bool,
        verbose: bool,
    ) -> None:
        if summary_df.empty:
            return

        rows = []
        for metric in self.config.metrics:
            for laser_idx in range(1, self.config.expected_lasers_count + 1):
                col = f"{metric}__Laser_{laser_idx}"
                if col not in summary_df.columns:
                    continue

                rows.append(
                    {
                        "metric": metric,
                        "laser": f"Laser_{laser_idx}",
                        "mean_value": float(summary_df[col].mean()),
                        "std_value": float(summary_df[col].std(ddof=1)) if len(summary_df[col]) >= 2 else np.nan,
                    }
                )

        laser_df = pd.DataFrame(rows)
        if laser_df.empty:
            return

        out_csv = out_dir / "laser_global_summary.csv"
        if write:
            laser_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        for metric, g in laser_df.groupby("metric", dropna=False):
            self._plot_bar(
                labels=g["laser"].tolist(),
                values=g["mean_value"].to_numpy(dtype=float),
                title=f"{metric} — media global por láser",
                ylabel=metric,
                save_path=out_dir / f"{metric}_global_by_laser.png",
                write=write,
            )

        if verbose:
            print("[CORR-CMP][OK] Per_Laser")

    # ======================================================
    # Global figures
    # ======================================================
    def _build_global_figures(
        self,
        global_summary_df: pd.DataFrame,
        pairwise_dates_summary_df: pd.DataFrame,
        pairwise_labs_summary_df: pd.DataFrame,
        out_dir: Path,
        *,
        write: bool,
    ) -> None:
        if global_summary_df.empty:
            return

        labels = [f"{f}|{l}" for f, l in zip(global_summary_df["fecha"], global_summary_df["lab"])]

        if "global_mean_anisotropy" in global_summary_df.columns:
            self._plot_bar(
                labels=labels,
                values=global_summary_df["global_mean_anisotropy"].to_numpy(dtype=float),
                title="Anisotropía media por bloque (fecha, lab)",
                ylabel="anisotropy",
                save_path=out_dir / "global_mean_anisotropy.png",
                write=write,
            )

        if "global_mean_pair_corr" in global_summary_df.columns:
            self._plot_bar(
                labels=labels,
                values=global_summary_df["global_mean_pair_corr"].to_numpy(dtype=float),
                title="Correlación media entre pares de láseres por bloque",
                ylabel="corr",
                save_path=out_dir / "global_mean_pair_corr.png",
                write=write,
            )

        if not pairwise_dates_summary_df.empty and "pairwise_profile_corr_mean" in pairwise_dates_summary_df.columns:
            labels_dates = [
                f"{lab}:{fa}-{fb}"
                for lab, fa, fb in zip(
                    pairwise_dates_summary_df["lab"],
                    pairwise_dates_summary_df["fecha_a"],
                    pairwise_dates_summary_df["fecha_b"],
                )
            ]
            self._plot_bar(
                labels=labels_dates,
                values=pairwise_dates_summary_df["pairwise_profile_corr_mean"].to_numpy(dtype=float),
                title="Correlación media entre fechas por laboratorio",
                ylabel="corr",
                save_path=out_dir / "pairwise_dates_profile_corr_mean.png",
                write=write,
            )

        if not pairwise_labs_summary_df.empty and "pairwise_profile_corr_mean" in pairwise_labs_summary_df.columns:
            labels_labs = [
                f"{fecha}:{la}-{lb}"
                for fecha, la, lb in zip(
                    pairwise_labs_summary_df["fecha"],
                    pairwise_labs_summary_df["lab_a"],
                    pairwise_labs_summary_df["lab_b"],
                )
            ]
            self._plot_bar(
                labels=labels_labs,
                values=pairwise_labs_summary_df["pairwise_profile_corr_mean"].to_numpy(dtype=float),
                title="Correlación media entre laboratorios por fecha",
                ylabel="corr",
                save_path=out_dir / "pairwise_labs_profile_corr_mean.png",
                write=write,
            )

    # ======================================================
    # Plot helpers
    # ======================================================
    def _plot_heatmap(
        self,
        *,
        matrix: np.ndarray,
        labels: Sequence[str],
        title: str,
        save_path: Path,
        write: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        if write:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_profile(
        self,
        *,
        values: np.ndarray,
        title: str,
        ylabel: str,
        save_path: Path,
        write: bool,
    ) -> None:
        x = np.arange(1, len(values) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("laser")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Laser_{i}" for i in x])
        fig.tight_layout()
        if write:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_bar(
        self,
        *,
        labels: Sequence[str],
        values: np.ndarray,
        title: str,
        ylabel: str,
        save_path: Path,
        write: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
        ax.bar(range(len(labels)), values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        if write:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ======================================================
    # Matrix helper
    # ======================================================
    def _pairwise_corr_to_matrix(self, sub: pd.DataFrame, *, expected_lasers_count: int) -> np.ndarray:
        n = expected_lasers_count
        mat = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(mat, 1.0)

        if sub.empty:
            return mat

        for _, row in sub.iterrows():
            li = _laser_label_to_index(row["laser_i"])
            lj = _laser_label_to_index(row["laser_j"])
            if li is None or lj is None:
                continue
            val = row.get("corr_value", np.nan)
            mat[li - 1, lj - 1] = val
            mat[lj - 1, li - 1] = val

        return mat


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def _laser_label_to_index(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    txt = str(value).strip()
    digits = "".join(ch for ch in txt if ch.isdigit())
    if not digits:
        return None
    try:
        out = int(digits)
    except ValueError:
        return None
    return out if out >= 1 else None


def _summary_profile_vector(row: pd.Series, metric: str, expected_lasers_count: int) -> np.ndarray:
    vals = []
    for idx in range(1, expected_lasers_count + 1):
        vals.append(row.get(f"{metric}__Laser_{idx}", np.nan))
    return np.asarray(vals, dtype=float)


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)