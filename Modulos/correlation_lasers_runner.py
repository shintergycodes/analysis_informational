from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from correlation_lasers_config import CorrelationLasersConfig


# ==========================================================
# Correlation Lasers — Runner
# ==========================================================
#
# Level 3 — Correlation Lasers
#
# Purpose
# -------
# Consume Quality artifacts and build the tabular analytical layer for:
# - canonical base at fecha × lab × mid × laser
# - profiles by measurement
# - profile summary by (fecha, lab)
# - intra-block laser correlations
# - pairwise date comparisons within lab
# - pairwise lab comparisons within date
#
# This module does NOT:
# - read raw experiment CSV files
# - depend on Ctrl/Exp semantics
# - use turno/jornada as structural axes
# - generate plots
#
# Primary upstream artifacts
# --------------------------
# - resultados_luces.csv            (required)
# - quality_scores_by_file.csv      (optional)
#
# ==========================================================


# ----------------------------------------------------------
# Public artifacts
# ----------------------------------------------------------
@dataclass(frozen=True)
class CorrelationLasersArtifacts:
    output_root: Path
    base_csv: Path
    profiles_by_measurement_csv: Path
    summary_by_date_lab_csv: Path
    correlation_by_date_lab_csv: Path
    pairwise_dates_by_lab_csv: Path
    pairwise_labs_by_date_csv: Path
    run_metadata_json: Optional[Path]


# ----------------------------------------------------------
# Runner
# ----------------------------------------------------------
class CorrelationLasersRunner:
    def __init__(
        self,
        root: Path,
        config: CorrelationLasersConfig,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.config.validate()

    # ======================================================
    # Public API
    # ======================================================
    def run(
        self,
        *,
        resultados_csv: Optional[Path] = None,
        quality_scores_by_file_csv: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        write: bool = True,
        verbose: bool = False,
    ) -> CorrelationLasersArtifacts:
        """
        Run the Correlation Lasers block.

        Parameters
        ----------
        resultados_csv:
            Optional explicit path to resultados_luces.csv.
            If omitted, it is resolved relative to `root`.
        quality_scores_by_file_csv:
            Optional explicit path to quality_scores_by_file.csv.
            If omitted, it is resolved relative to `root`.
        output_dir:
            Optional explicit output directory.
            If omitted, defaults to `root / config.output_dir_name`.
        write:
            If True, write outputs to disk.
        verbose:
            If True, print informative progress messages.

        Returns
        -------
        CorrelationLasersArtifacts
        """
        resultados_path = self._resolve_resultados_csv(resultados_csv)
        quality_scores_path = self._resolve_quality_scores_csv(quality_scores_by_file_csv)
        output_root = self._resolve_output_dir(output_dir)

        if verbose:
            print(f"[CORR] resultados_csv           : {resultados_path}")
            print(f"[CORR] quality_scores_by_file   : {quality_scores_path}")
            print(f"[CORR] output_root              : {output_root}")

        df_raw = self._read_csv_required(resultados_path)
        df_base = self._build_canonical_base(df_raw)

        if verbose:
            print(f"[CORR] base rows after canonicalization: {len(df_base)}")

        scores_df = None
        if quality_scores_path is not None and quality_scores_path.exists():
            scores_df = self._read_csv_optional(quality_scores_path)
            if scores_df is not None and verbose:
                print(f"[CORR] quality_scores_by_file rows     : {len(scores_df)}")

        profiles_df = self._build_profiles_by_measurement(df_base)
        profiles_df = self._attach_quality_scores(profiles_df, scores_df)

        summary_df = self._build_summary_by_date_lab(df_base)
        corr_df = self._build_intra_block_correlation(df_base)
        pair_dates_df = self._build_pairwise_dates_by_lab(summary_df)
        pair_labs_df = self._build_pairwise_labs_by_date(summary_df)

        if verbose:
            print(f"[CORR] profiles_by_measurement rows    : {len(profiles_df)}")
            print(f"[CORR] summary_by_date_lab rows        : {len(summary_df)}")
            print(f"[CORR] correlation_by_date_lab rows    : {len(corr_df)}")
            print(f"[CORR] pairwise_dates_by_lab rows      : {len(pair_dates_df)}")
            print(f"[CORR] pairwise_labs_by_date rows      : {len(pair_labs_df)}")

        metadata_path: Optional[Path] = None
        artifacts = CorrelationLasersArtifacts(
            output_root=output_root,
            base_csv=output_root / self.config.artifacts.base_csv,
            profiles_by_measurement_csv=output_root / self.config.artifacts.profiles_by_measurement_csv,
            summary_by_date_lab_csv=output_root / self.config.artifacts.summary_by_date_lab_csv,
            correlation_by_date_lab_csv=output_root / self.config.artifacts.correlation_by_date_lab_csv,
            pairwise_dates_by_lab_csv=output_root / self.config.artifacts.pairwise_dates_by_lab_csv,
            pairwise_labs_by_date_csv=output_root / self.config.artifacts.pairwise_labs_by_date_csv,
            run_metadata_json=(
                output_root / self.config.artifacts.run_metadata_json
                if self.config.write_metadata_json
                else None
            ),
        )

        if write:
            output_root.mkdir(parents=True, exist_ok=True)

            df_base.to_csv(artifacts.base_csv, index=False, encoding="utf-8-sig")
            profiles_df.to_csv(artifacts.profiles_by_measurement_csv, index=False, encoding="utf-8-sig")
            summary_df.to_csv(artifacts.summary_by_date_lab_csv, index=False, encoding="utf-8-sig")
            corr_df.to_csv(artifacts.correlation_by_date_lab_csv, index=False, encoding="utf-8-sig")
            pair_dates_df.to_csv(artifacts.pairwise_dates_by_lab_csv, index=False, encoding="utf-8-sig")
            pair_labs_df.to_csv(artifacts.pairwise_labs_by_date_csv, index=False, encoding="utf-8-sig")

            if artifacts.run_metadata_json is not None:
                metadata = self._build_run_metadata(
                    resultados_path=resultados_path,
                    quality_scores_path=quality_scores_path,
                    output_root=output_root,
                    df_raw=df_raw,
                    df_base=df_base,
                    profiles_df=profiles_df,
                    summary_df=summary_df,
                    corr_df=corr_df,
                    pair_dates_df=pair_dates_df,
                    pair_labs_df=pair_labs_df,
                )
                artifacts.run_metadata_json.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                metadata_path = artifacts.run_metadata_json

            if verbose:
                print(f"[CORR][OK] base_csv                  : {artifacts.base_csv}")
                print(f"[CORR][OK] profiles_by_measurement  : {artifacts.profiles_by_measurement_csv}")
                print(f"[CORR][OK] summary_by_date_lab      : {artifacts.summary_by_date_lab_csv}")
                print(f"[CORR][OK] correlation_by_date_lab  : {artifacts.correlation_by_date_lab_csv}")
                print(f"[CORR][OK] pairwise_dates_by_lab    : {artifacts.pairwise_dates_by_lab_csv}")
                print(f"[CORR][OK] pairwise_labs_by_date    : {artifacts.pairwise_labs_by_date_csv}")
                if metadata_path is not None:
                    print(f"[CORR][OK] run_metadata_json        : {metadata_path}")

        return artifacts

    # ======================================================
    # Path resolution
    # ======================================================
    def _resolve_resultados_csv(self, explicit: Optional[Path]) -> Path:
        if explicit is not None:
            return Path(explicit)

        candidates = [
            self.root / self.config.artifacts.input_resultados_luces_csv,
            self.root / "Reports" / "Level1_Quality" / self.config.artifacts.input_resultados_luces_csv,
        ]
        for p in candidates:
            if p.exists():
                return p

        return candidates[-1]

    def _resolve_quality_scores_csv(self, explicit: Optional[Path]) -> Optional[Path]:
        if explicit is not None:
            return Path(explicit)

        candidates = [
            self.root / self.config.artifacts.input_quality_scores_by_file_csv,
            self.root / "Reports" / "Level1_Quality" / self.config.artifacts.input_quality_scores_by_file_csv,
        ]
        for p in candidates:
            if p.exists():
                return p

        return candidates[-1]

    def _resolve_output_dir(self, explicit: Optional[Path]) -> Path:
        if explicit is not None:
            return Path(explicit)
        return self.root / self.config.output_dir_name

    # ======================================================
    # Reads
    # ======================================================
    def _read_csv_required(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Required Correlation Lasers input not found: {path}")
        return pd.read_csv(path)

    def _read_csv_optional(self, path: Optional[Path]) -> Optional[pd.DataFrame]:
        if path is None or not path.exists():
            return None
        return pd.read_csv(path)

    # ======================================================
    # Base canonicalization
    # ======================================================
    def _build_canonical_base(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        schema = self.config.input_schema
        df = df_raw.copy()

        # Normalize laser column name to internal canonical `laser`
        if schema.laser_col not in df.columns:
            if "laser" in df.columns:
                df[schema.laser_col] = df["laser"]
            else:
                raise ValueError(
                    f"Primary input is missing laser column. "
                    f"Expected '{schema.laser_col}' or 'laser'."
                )

        missing = [c for c in schema.required_columns() if c not in df.columns]
        if missing:
            raise ValueError(
                "Primary input is missing required columns: "
                f"{missing}. Required: {list(schema.required_columns())}"
            )

        rename_map = {
            schema.fecha_col: "fecha",
            schema.lab_col: "lab",
            schema.mid_col: "mid",
            schema.laser_col: "laser",
            schema.calidad_col: "calidad_general",
            schema.snr_col: "snr_db",
            schema.cv_col: "coef_variacion",
        }

        optional_map = {
            schema.media_col: "media",
            schema.desviacion_col: "desviacion",
            schema.ruido_rms_col: "ruido_rms",
            schema.tendencia_col: "tendencia_por_muestra",
            schema.num_muestras_col: "num_muestras",
            schema.valido_col: "valido",
            schema.status_col: "status",
        }

        for src, dst in optional_map.items():
            if src in df.columns:
                rename_map[src] = dst

        df = df.rename(columns=rename_map)

        keep_cols = list(dict.fromkeys(rename_map.values()))
        df = df[keep_cols].copy()

        # Normalize structural columns
        for col in ("fecha", "lab", "mid", "laser"):
            df[col] = df[col].astype(str).str.strip()

        # Normalize metrics to numeric
        numeric_cols = [
            "calidad_general",
            "snr_db",
            "coef_variacion",
            "media",
            "desviacion",
            "ruido_rms",
            "tendencia_por_muestra",
            "num_muestras",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalize validity columns if present
        if "valido" in df.columns:
            df["valido"] = self._normalize_valido(df["valido"])

        if "status" in df.columns:
            df["status"] = df["status"].astype(str).str.strip()

        # Optional drop invalid rows
        if self.config.drop_invalid_rows:
            df = self._filter_invalid_rows(df)

        # Canonical laser index and label
        df["laser"] = df["laser"].astype(str).str.strip()
        df["laser_index"] = df["laser"].map(_extract_laser_index)
        df["laser_ok"] = df["laser_index"].notna()

        # Remove rows without valid laser index
        df = df[df["laser_ok"]].copy()
        df["laser_index"] = df["laser_index"].astype(int)
        df["laser_key"] = df["laser_index"].apply(lambda x: f"Laser_{x}")

        # Drop exact duplicates conservatively on structural identity
        df = (
            df.sort_values(["fecha", "lab", "mid", "laser_index"])
            .drop_duplicates(subset=["fecha", "lab", "mid", "laser_index"], keep="first")
            .reset_index(drop=True)
        )

        # Count valid lasers per measurement
        laser_counts = (
            df.groupby(["fecha", "lab", "mid"], dropna=False)["laser_index"]
            .nunique()
            .rename("valid_lasers_count")
            .reset_index()
        )
        df = df.merge(laser_counts, on=["fecha", "lab", "mid"], how="left")

        df["measurement_laser_count_ok"] = (
            df["valid_lasers_count"] >= self.config.min_valid_lasers_per_measurement
        )

        if self.config.require_all_6_lasers:
            df = df[df["valid_lasers_count"] == self.config.expected_lasers_count].copy()
        else:
            df = df[df["measurement_laser_count_ok"]].copy()

        # Reorder
        preferred_order = [
            "fecha",
            "lab",
            "mid",
            "laser",
            "laser_key",
            "laser_index",
            "calidad_general",
            "snr_db",
            "coef_variacion",
            "media",
            "desviacion",
            "ruido_rms",
            "tendencia_por_muestra",
            "num_muestras",
            "valido",
            "status",
            "laser_ok",
            "valid_lasers_count",
            "measurement_laser_count_ok",
        ]
        ordered_cols = [c for c in preferred_order if c in df.columns]
        remaining_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + remaining_cols].copy()

        return df

    def _normalize_valido(self, s: pd.Series) -> pd.Series:
        def convert(v: object) -> bool:
            if isinstance(v, bool):
                return v
            if pd.isna(v):
                return False
            txt = str(v).strip().lower()
            return txt in {"true", "1", "yes", "y", "si", "sí", "ok", "valid", "valido", "válido"}

        return s.map(convert).astype(bool)

    def _filter_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "valido" in out.columns:
            out = out[out["valido"].isin(self.config.valid_boolean_values)].copy()

        if "status" in out.columns and self.config.keep_status_column:
            valid_status_lower = {str(x).strip().lower() for x in self.config.valid_status_values}
            out = out[out["status"].astype(str).str.strip().str.lower().isin(valid_status_lower)].copy()

        # Require non-null primary metrics
        for metric in self.config.metrics:
            if metric in out.columns:
                out = out[out[metric].notna()].copy()

        return out

    # ======================================================
    # Profiles by measurement
    # ======================================================
    def _build_profiles_by_measurement(self, df_base: pd.DataFrame) -> pd.DataFrame:
        if df_base.empty:
            return self._empty_profiles_df()

        metric_cols = [m for m in self.config.metrics if m in df_base.columns]
        if not metric_cols:
            raise ValueError("None of the configured metrics are available in the canonical base table.")

        # block means by (fecha, lab, laser_index)
        block_means = (
            df_base.groupby(["fecha", "lab", "laser_index"], dropna=False)[metric_cols]
            .agg(self.config.aggregate_mode)
            .reset_index()
        )
        block_means = block_means.rename(columns={m: f"block_mean__{m}" for m in metric_cols})

        merged = df_base.merge(
            block_means,
            on=["fecha", "lab", "laser_index"],
            how="left",
        )

        rows: List[Dict[str, object]] = []

        for (fecha, lab, mid), g in merged.groupby(["fecha", "lab", "mid"], dropna=False):
            g = g.sort_values("laser_index").copy()

            row: Dict[str, object] = {
                "fecha": fecha,
                "lab": lab,
                "mid": mid,
                "n_lasers_present": int(g["laser_index"].nunique()),
                "lasers_present": ",".join(g["laser_key"].astype(str).tolist()),
            }

            for metric in metric_cols:
                values = g[metric].to_numpy(dtype=float)
                block_values = g[f"block_mean__{metric}"].to_numpy(dtype=float)

                # per-laser columns
                for _, rr in g.iterrows():
                    row[f"{metric}__{rr['laser_key']}"] = _safe_float(rr[metric])

                row[f"{metric}__profile_mean"] = _safe_float(np.nanmean(values))
                row[f"{metric}__profile_std"] = _safe_float(np.nanstd(values, ddof=1)) if len(values) >= 2 else np.nan
                row[f"{metric}__profile_min"] = _safe_float(np.nanmin(values))
                row[f"{metric}__profile_max"] = _safe_float(np.nanmax(values))
                row[f"{metric}__anisotropy_ratio"] = _ratio_max_min(values)
                row[f"{metric}__coherence_corr"] = _corr(values, block_values, self.config.corr_method)
                row[f"{metric}__distance_to_block"] = _euclidean_distance(values, block_values)

            if self.config.compute_profile_coherence:
                coherence_values = [
                    row.get(f"{metric}__coherence_corr")
                    for metric in metric_cols
                    if pd.notna(row.get(f"{metric}__coherence_corr"))
                ]
                row["profile_coherence_score"] = (
                    _safe_float(np.nanmean(coherence_values)) if coherence_values else np.nan
                )

            if self.config.compute_profile_distances:
                distance_values = [
                    row.get(f"{metric}__distance_to_block")
                    for metric in metric_cols
                    if pd.notna(row.get(f"{metric}__distance_to_block"))
                ]
                row["profile_distance_score"] = (
                    _safe_float(np.nanmean(distance_values)) if distance_values else np.nan
                )

            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return self._empty_profiles_df()

        return df.sort_values(["fecha", "lab", "mid"]).reset_index(drop=True)

    def _attach_quality_scores(
        self,
        profiles_df: pd.DataFrame,
        scores_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if scores_df is None or profiles_df.empty:
            return profiles_df

        candidate_mid_cols = ["mid", "MID"]
        score_mid_col = next((c for c in candidate_mid_cols if c in scores_df.columns), None)
        if score_mid_col is None:
            return profiles_df

        score_cols_preferred = [
            "quality_score",
            "score_global",
            "score_total",
            "calidad_general_archivo",
            "calidad_archivo",
            "pass_score",
        ]
        keep = [score_mid_col] + [c for c in score_cols_preferred if c in scores_df.columns]
        keep = list(dict.fromkeys(keep))

        if len(keep) <= 1:
            return profiles_df

        aux = scores_df[keep].copy().rename(columns={score_mid_col: "mid"})
        aux["mid"] = aux["mid"].astype(str).str.strip()

        return profiles_df.merge(aux, on="mid", how="left")

    def _empty_profiles_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["fecha", "lab", "mid"])

    # ======================================================
    # Summary by (fecha, lab)
    # ======================================================
    def _build_summary_by_date_lab(self, df_base: pd.DataFrame) -> pd.DataFrame:
        if df_base.empty:
            return pd.DataFrame(columns=["fecha", "lab"])

        metric_cols = [m for m in self.config.metrics if m in df_base.columns]
        rows: List[Dict[str, object]] = []

        for (fecha, lab), g in df_base.groupby(["fecha", "lab"], dropna=False):
            row: Dict[str, object] = {
                "fecha": fecha,
                "lab": lab,
                "n_measurements": int(g["mid"].nunique()),
                "n_rows": int(len(g)),
                "n_unique_lasers": int(g["laser_index"].nunique()),
                "block_valid_for_pairwise": bool(
                    g["mid"].nunique() >= self.config.min_measurements_per_block
                ),
            }

            # Means by laser for each metric
            per_laser = (
                g.groupby("laser_index", dropna=False)[metric_cols]
                .agg(self.config.aggregate_mode)
                .reset_index()
                .sort_values("laser_index")
            )

            for metric in metric_cols:
                vec = per_laser[metric].to_numpy(dtype=float)

                for _, rr in per_laser.iterrows():
                    laser_key = f"Laser_{int(rr['laser_index'])}"
                    row[f"{metric}__{laser_key}"] = _safe_float(rr[metric])

                row[f"{metric}__profile_mean"] = _safe_float(np.nanmean(vec))
                row[f"{metric}__profile_std"] = _safe_float(np.nanstd(vec, ddof=1)) if len(vec) >= 2 else np.nan
                row[f"{metric}__profile_min"] = _safe_float(np.nanmin(vec))
                row[f"{metric}__profile_max"] = _safe_float(np.nanmax(vec))
                row[f"{metric}__anisotropy_ratio"] = _ratio_max_min(vec)

            # Internal consistency across measurements per laser
            for metric in metric_cols:
                tmp = (
                    g.groupby(["mid", "laser_index"], dropna=False)[metric]
                    .agg(self.config.aggregate_mode)
                    .reset_index()
                )
                laser_stds = (
                    tmp.groupby("laser_index", dropna=False)[metric]
                    .std(ddof=1)
                    .rename("laser_std")
                    .reset_index()
                )
                row[f"{metric}__intermeasurement_std_mean"] = _safe_float(
                    laser_stds["laser_std"].mean()
                )

            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["fecha", "lab"])

        return out.sort_values(["fecha", "lab"]).reset_index(drop=True)

    # ======================================================
    # Intra-block correlation by (fecha, lab)
    # ======================================================
    def _build_intra_block_correlation(self, df_base: pd.DataFrame) -> pd.DataFrame:
        if not self.config.compute_intra_block_correlation or df_base.empty:
            return pd.DataFrame(
                columns=[
                    "fecha",
                    "lab",
                    "metric",
                    "laser_i",
                    "laser_j",
                    "corr_method",
                    "corr_value",
                    "n_pairs",
                    "block_n_measurements",
                    "valid_for_correlation",
                ]
            )

        metric_cols = [m for m in self.config.metrics if m in df_base.columns]
        rows: List[Dict[str, object]] = []

        for (fecha, lab), g in df_base.groupby(["fecha", "lab"], dropna=False):
            n_meas = int(g["mid"].nunique())

            for metric in metric_cols:
                pivot = (
                    g.pivot_table(
                        index="mid",
                        columns="laser_index",
                        values=metric,
                        aggfunc=self.config.aggregate_mode,
                    )
                    .sort_index(axis=1)
                )

                available_lasers = [int(c) for c in pivot.columns if pd.notna(c)]

                for li, lj in combinations(available_lasers, 2):
                    s1 = pivot[li]
                    s2 = pivot[lj]
                    valid = s1.notna() & s2.notna()
                    n_pairs = int(valid.sum())

                    corr_value = np.nan
                    valid_corr = False

                    if n_pairs >= 2:
                        corr_value = _corr(
                            s1[valid].to_numpy(dtype=float),
                            s2[valid].to_numpy(dtype=float),
                            self.config.corr_method,
                        )
                        valid_corr = pd.notna(corr_value)

                    rows.append(
                        {
                            "fecha": fecha,
                            "lab": lab,
                            "metric": metric,
                            "laser_i": f"Laser_{li}",
                            "laser_j": f"Laser_{lj}",
                            "corr_method": self.config.corr_method,
                            "corr_value": _safe_float(corr_value),
                            "n_pairs": n_pairs,
                            "block_n_measurements": n_meas,
                            "valid_for_correlation": bool(valid_corr),
                        }
                    )

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "fecha",
                    "lab",
                    "metric",
                    "laser_i",
                    "laser_j",
                    "corr_method",
                    "corr_value",
                    "n_pairs",
                    "block_n_measurements",
                    "valid_for_correlation",
                ]
            )

        return out.sort_values(["fecha", "lab", "metric", "laser_i", "laser_j"]).reset_index(drop=True)

    # ======================================================
    # Pairwise dates by lab
    # ======================================================
    def _build_pairwise_dates_by_lab(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.compute_pairwise_dates_by_lab or summary_df.empty:
            return pd.DataFrame(columns=["lab", "fecha_a", "fecha_b"])

        metric_cols = [m for m in self.config.metrics]
        rows: List[Dict[str, object]] = []

        for lab, g in summary_df.groupby("lab", dropna=False):
            g = g.sort_values("fecha").reset_index(drop=True)

            if len(g) < 2:
                continue

            for idx_a, idx_b in combinations(range(len(g)), 2):
                a = g.iloc[idx_a]
                b = g.iloc[idx_b]

                if self.config.require_min_measurements_for_pairwise:
                    if not bool(a.get("block_valid_for_pairwise", False)):
                        continue
                    if not bool(b.get("block_valid_for_pairwise", False)):
                        continue

                row: Dict[str, object] = {
                    "lab": lab,
                    "fecha_a": a["fecha"],
                    "fecha_b": b["fecha"],
                    "n_measurements_a": int(a.get("n_measurements", 0)),
                    "n_measurements_b": int(b.get("n_measurements", 0)),
                }

                pair_corrs: List[float] = []
                pair_dists: List[float] = []

                for metric in metric_cols:
                    vec_a = _extract_profile_vector(a, metric, self.config.expected_lasers_count)
                    vec_b = _extract_profile_vector(b, metric, self.config.expected_lasers_count)

                    corr_val = (
                        _corr(vec_a, vec_b, self.config.corr_method)
                        if self.config.compute_pairwise_profile_correlations
                        else np.nan
                    )
                    dist_val = (
                        _euclidean_distance(vec_a, vec_b)
                        if self.config.compute_profile_distances
                        else np.nan
                    )
                    anis_a = _safe_float(a.get(f"{metric}__anisotropy_ratio", np.nan))
                    anis_b = _safe_float(b.get(f"{metric}__anisotropy_ratio", np.nan))

                    row[f"{metric}__profile_corr"] = _safe_float(corr_val)
                    row[f"{metric}__profile_distance"] = _safe_float(dist_val)
                    row[f"{metric}__anisotropy_ratio_a"] = anis_a
                    row[f"{metric}__anisotropy_ratio_b"] = anis_b
                    row[f"{metric}__anisotropy_delta"] = (
                        _safe_float(anis_b - anis_a)
                        if pd.notna(anis_a) and pd.notna(anis_b)
                        else np.nan
                    )

                    if pd.notna(corr_val):
                        pair_corrs.append(float(corr_val))
                    if pd.notna(dist_val):
                        pair_dists.append(float(dist_val))

                row["pairwise_profile_corr_mean"] = (
                    _safe_float(np.nanmean(pair_corrs)) if pair_corrs else np.nan
                )
                row["pairwise_profile_distance_mean"] = (
                    _safe_float(np.nanmean(pair_dists)) if pair_dists else np.nan
                )

                rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["lab", "fecha_a", "fecha_b"])

        return out.sort_values(["lab", "fecha_a", "fecha_b"]).reset_index(drop=True)

    # ======================================================
    # Pairwise labs by date
    # ======================================================
    def _build_pairwise_labs_by_date(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.compute_pairwise_labs_by_date or summary_df.empty:
            return pd.DataFrame(columns=["fecha", "lab_a", "lab_b"])

        metric_cols = [m for m in self.config.metrics]
        rows: List[Dict[str, object]] = []

        for fecha, g in summary_df.groupby("fecha", dropna=False):
            g = g.sort_values("lab").reset_index(drop=True)

            if len(g) < 2:
                continue

            for idx_a, idx_b in combinations(range(len(g)), 2):
                a = g.iloc[idx_a]
                b = g.iloc[idx_b]

                if self.config.require_min_measurements_for_pairwise:
                    if not bool(a.get("block_valid_for_pairwise", False)):
                        continue
                    if not bool(b.get("block_valid_for_pairwise", False)):
                        continue

                row: Dict[str, object] = {
                    "fecha": fecha,
                    "lab_a": a["lab"],
                    "lab_b": b["lab"],
                    "n_measurements_a": int(a.get("n_measurements", 0)),
                    "n_measurements_b": int(b.get("n_measurements", 0)),
                }

                pair_corrs: List[float] = []
                pair_dists: List[float] = []

                for metric in metric_cols:
                    vec_a = _extract_profile_vector(a, metric, self.config.expected_lasers_count)
                    vec_b = _extract_profile_vector(b, metric, self.config.expected_lasers_count)

                    corr_val = (
                        _corr(vec_a, vec_b, self.config.corr_method)
                        if self.config.compute_pairwise_profile_correlations
                        else np.nan
                    )
                    dist_val = (
                        _euclidean_distance(vec_a, vec_b)
                        if self.config.compute_profile_distances
                        else np.nan
                    )
                    anis_a = _safe_float(a.get(f"{metric}__anisotropy_ratio", np.nan))
                    anis_b = _safe_float(b.get(f"{metric}__anisotropy_ratio", np.nan))

                    row[f"{metric}__profile_corr"] = _safe_float(corr_val)
                    row[f"{metric}__profile_distance"] = _safe_float(dist_val)
                    row[f"{metric}__anisotropy_ratio_a"] = anis_a
                    row[f"{metric}__anisotropy_ratio_b"] = anis_b
                    row[f"{metric}__anisotropy_delta"] = (
                        _safe_float(anis_b - anis_a)
                        if pd.notna(anis_a) and pd.notna(anis_b)
                        else np.nan
                    )

                    if pd.notna(corr_val):
                        pair_corrs.append(float(corr_val))
                    if pd.notna(dist_val):
                        pair_dists.append(float(dist_val))

                row["pairwise_profile_corr_mean"] = (
                    _safe_float(np.nanmean(pair_corrs)) if pair_corrs else np.nan
                )
                row["pairwise_profile_distance_mean"] = (
                    _safe_float(np.nanmean(pair_dists)) if pair_dists else np.nan
                )

                rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["fecha", "lab_a", "lab_b"])

        return out.sort_values(["fecha", "lab_a", "lab_b"]).reset_index(drop=True)

    # ======================================================
    # Metadata
    # ======================================================
    def _build_run_metadata(
        self,
        *,
        resultados_path: Path,
        quality_scores_path: Optional[Path],
        output_root: Path,
        df_raw: pd.DataFrame,
        df_base: pd.DataFrame,
        profiles_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        corr_df: pd.DataFrame,
        pair_dates_df: pd.DataFrame,
        pair_labs_df: pd.DataFrame,
    ) -> Dict[str, object]:
        return {
            "block_name": self.config.block_name,
            "version": self.config.version,
            "root": str(self.root),
            "input_resultados_csv": str(resultados_path),
            "input_quality_scores_csv": str(quality_scores_path) if quality_scores_path is not None else None,
            "output_root": str(output_root),
            "config_summary": self.config.summary(),
            "row_counts": {
                "raw_input_rows": int(len(df_raw)),
                "base_rows": int(len(df_base)),
                "profiles_by_measurement_rows": int(len(profiles_df)),
                "summary_by_date_lab_rows": int(len(summary_df)),
                "correlation_by_date_lab_rows": int(len(corr_df)),
                "pairwise_dates_by_lab_rows": int(len(pair_dates_df)),
                "pairwise_labs_by_date_rows": int(len(pair_labs_df)),
            },
            "distinct_counts": {
                "fechas": int(df_base["fecha"].nunique()) if not df_base.empty else 0,
                "labs": int(df_base["lab"].nunique()) if not df_base.empty else 0,
                "mids": int(df_base["mid"].nunique()) if not df_base.empty else 0,
                "lasers": int(df_base["laser_index"].nunique()) if not df_base.empty else 0,
            },
        }


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def _extract_laser_index(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    txt = str(value).strip()
    digits = "".join(ch for ch in txt if ch.isdigit())
    if not digits:
        return None
    try:
        idx = int(digits)
    except ValueError:
        return None
    return idx if idx >= 1 else None


def _safe_float(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _ratio_max_min(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.nan
    if vmin == 0:
        return np.nan
    return _safe_float(vmax / vmin)


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(xb)
    if valid.sum() == 0:
        return np.nan
    return _safe_float(np.linalg.norm(xa[valid] - xb[valid]))


def _corr(a: Sequence[float], b: Sequence[float], method: str) -> float:
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(xb)

    if valid.sum() < 2:
        return np.nan

    xa = xa[valid]
    xb = xb[valid]

    if np.nanstd(xa) == 0 or np.nanstd(xb) == 0:
        return np.nan

    if method == "pearson":
        return _safe_float(np.corrcoef(xa, xb)[0, 1])

    if method == "spearman":
        ra = pd.Series(xa).rank(method="average").to_numpy(dtype=float)
        rb = pd.Series(xb).rank(method="average").to_numpy(dtype=float)
        return _safe_float(np.corrcoef(ra, rb)[0, 1])

    raise ValueError(f"Unsupported correlation method: {method}")


def _extract_profile_vector(
    row: pd.Series,
    metric: str,
    expected_lasers_count: int,
) -> np.ndarray:
    vals = []
    for idx in range(1, expected_lasers_count + 1):
        vals.append(row.get(f"{metric}__Laser_{idx}", np.nan))
    return np.asarray(vals, dtype=float)