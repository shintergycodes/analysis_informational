from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


# ==========================================================
# Correlation Lasers — Config Contract
# ==========================================================
#
# Purpose
# -------
# Declarative contract for Level 3: Correlation Lasers.
#
# This module does NOT:
# - read experimental CSV files
# - compute correlations
# - generate plots
#
# This module DOES:
# - define the analytical contract of the block
# - define required inputs from Quality
# - define enabled metrics / policies
# - define official output artifact names
# - validate internal consistency
#
# Active structural contract
# --------------------------
# fecha -> lab -> medición
#
# Active axes:
# - fecha
# - lab
# - mid
#
# Internal fixed substructure:
# - laser (expected dimension: 6)
#
# Non-active / optional legacy metadata:
# - jornada
# - turno
# - group
# - label
#
# Primary upstream dependency
# ---------------------------
# The block is designed to consume Quality artifacts, mainly:
# - resultados_luces.csv
# - quality_scores_by_file.csv   (optional helper)
#
# ==========================================================


# ----------------------------------------------------------
# Allowed enums / constants
# ----------------------------------------------------------
ALLOWED_METRICS = {
    "calidad_general",
    "snr_db",
    "coef_variacion",
    "media",
    "desviacion",
    "ruido_rms",
    "tendencia_por_muestra",
}

ALLOWED_CORR_METHODS = {
    "pearson",
    "spearman",
}

ALLOWED_AGGREGATE_MODES = {
    "mean",
    "median",
}


# ----------------------------------------------------------
# Official artifact names
# ----------------------------------------------------------
@dataclass(frozen=True)
class CorrelationArtifacts:
    """
    Official output artifact names for Correlation Lasers.

    Notes
    -----
    These are file/directory names only, not absolute paths.
    Path resolution is handled by the runner / orchestrator.
    """

    # Inputs (upstream Quality artifacts)
    input_resultados_luces_csv: str = "resultados_luces.csv"
    input_quality_scores_by_file_csv: str = "quality_scores_by_file.csv"

    # Core runner outputs
    base_csv: str = "correlation_lasers_base.csv"
    profiles_by_measurement_csv: str = "laser_profiles_by_measurement.csv"
    summary_by_date_lab_csv: str = "laser_profile_summary_by_date_lab.csv"
    correlation_by_date_lab_csv: str = "laser_correlation_by_date_lab.csv"
    pairwise_dates_by_lab_csv: str = "laser_pairwise_dates_by_lab.csv"
    pairwise_labs_by_date_csv: str = "laser_pairwise_labs_by_date.csv"
    run_metadata_json: str = "correlation_lasers_run_metadata.json"

    # Compare / reporting outputs
    reports_root_dir: str = "Reports/Level3_Correlation_Lasers"
    global_dir: str = "comparacion_global"
    per_date_lab_dir: str = "Per_Date_Lab"
    per_laser_dir: str = "Per_Laser"
    per_lab_dir: str = "Per_Lab"


# ----------------------------------------------------------
# Input schema contract
# ----------------------------------------------------------
@dataclass(frozen=True)
class CorrelationInputSchema:
    """
    Canonical expected columns for the primary input table.

    The runner may internally normalize aliases (e.g. laser_name -> laser),
    but this schema expresses the intended canonical contract.
    """

    fecha_col: str = "fecha"
    lab_col: str = "lab"
    mid_col: str = "mid"
    laser_col: str = "laser_name"

    # Primary metric columns inherited from Quality
    calidad_col: str = "calidad_general"
    snr_col: str = "snr_db"
    cv_col: str = "coef_variacion"

    # Optional but strongly useful auxiliary columns
    media_col: str = "media"
    desviacion_col: str = "desviacion"
    ruido_rms_col: str = "ruido_rms"
    tendencia_col: str = "tendencia_por_muestra"
    num_muestras_col: str = "num_muestras"
    valido_col: str = "valido"
    status_col: str = "status"

    def required_columns(self) -> Tuple[str, ...]:
        """
        Minimum required columns for the primary input artifact.
        """
        return (
            self.fecha_col,
            self.lab_col,
            self.mid_col,
            self.laser_col,
            self.calidad_col,
            self.snr_col,
            self.cv_col,
        )

    def useful_optional_columns(self) -> Tuple[str, ...]:
        """
        Helpful optional columns if present.
        """
        return (
            self.media_col,
            self.desviacion_col,
            self.ruido_rms_col,
            self.tendencia_col,
            self.num_muestras_col,
            self.valido_col,
            self.status_col,
        )


# ----------------------------------------------------------
# Core config
# ----------------------------------------------------------
@dataclass
class CorrelationLasersConfig:
    """
    Declarative configuration for Level 3 — Correlation Lasers.

    Design goals
    ------------
    - Multi-lab compatible
    - Blind-compatible
    - No Ctrl/Exp semantics
    - No turno/jornada as structural axes
    - Deterministic and auditable

    Main analytical responsibilities enabled by this config
    -------------------------------------------------------
    1) Canonicalize Quality outputs at:
           fecha × lab × mid × laser
    2) Build per-measurement laser profiles
    3) Build per-(fecha, lab) mean profiles
    4) Compute intra-block laser correlations
    5) Compute coherence of each measurement against its block profile
    6) Compare profiles across dates within lab
    7) Compare profiles across labs within date
    """

    # ----------------------------
    # Identity
    # ----------------------------
    block_name: str = "Correlation Lasers"
    version: str = "1.0.0"

    # ----------------------------
    # Metrics
    # ----------------------------
    metrics: Tuple[str, ...] = (
        "calidad_general",
        "snr_db",
        "coef_variacion",
    )

    # Correlation and aggregation policies
    corr_method: str = "pearson"
    aggregate_mode: str = "mean"

    # ----------------------------
    # Structural expectations
    # ----------------------------
    expected_lasers_count: int = 6
    require_all_6_lasers: bool = True

    # ----------------------------
    # Validity / filtering
    # ----------------------------
    drop_invalid_rows: bool = True
    keep_status_column: bool = True
    valid_status_values: Tuple[str, ...] = ("ok", "OK", "valid", "VALID")
    valid_boolean_values: Tuple[bool, ...] = (True,)

    min_measurements_per_block: int = 3
    min_valid_lasers_per_measurement: int = 4

    # If True, pairwise block comparisons only run when both blocks
    # satisfy min_measurements_per_block.
    require_min_measurements_for_pairwise: bool = True

    # ----------------------------
    # Enabled analytical stages
    # ----------------------------
    compute_profiles_by_measurement: bool = True
    compute_summary_by_date_lab: bool = True
    compute_intra_block_correlation: bool = True
    compute_profile_coherence: bool = True
    compute_pairwise_dates_by_lab: bool = True
    compute_pairwise_labs_by_date: bool = True

    # ----------------------------
    # Derived diagnostics
    # ----------------------------
    compute_anisotropy_indices: bool = True
    compute_profile_distances: bool = True
    compute_pairwise_profile_correlations: bool = True

    # ----------------------------
    # Output policy
    # ----------------------------
    output_dir_name: str = "Reports/Level3_Correlation_Lasers"
    overwrite: bool = True
    write_metadata_json: bool = True

    # ----------------------------
    # Upstream / downstream artifact contracts
    # ----------------------------
    artifacts: CorrelationArtifacts = field(default_factory=CorrelationArtifacts)
    input_schema: CorrelationInputSchema = field(default_factory=CorrelationInputSchema)

    # ----------------------------
    # Optional metadata
    # ----------------------------
    notes: str = (
        "Blind-native relational analysis of the 6-laser system using "
        "Quality outputs only."
    )

    # ======================================================
    # Validation
    # ======================================================
    def validate(self) -> None:
        """
        Validate internal consistency of the configuration.
        Raises ValueError if the contract is inconsistent.
        """
        self._validate_identity()
        self._validate_metrics()
        self._validate_policies()
        self._validate_structure()
        self._validate_enabled_stages()
        self._validate_artifacts()
        self._validate_input_schema()

    def _validate_identity(self) -> None:
        if not self.block_name.strip():
            raise ValueError("block_name cannot be empty.")
        if not self.version.strip():
            raise ValueError("version cannot be empty.")

    def _validate_metrics(self) -> None:
        if not self.metrics:
            raise ValueError("metrics cannot be empty.")

        unknown = [m for m in self.metrics if m not in ALLOWED_METRICS]
        if unknown:
            raise ValueError(
                f"Unknown metrics in config: {unknown}. "
                f"Allowed metrics: {sorted(ALLOWED_METRICS)}"
            )

        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics contains duplicates.")

        if self.corr_method not in ALLOWED_CORR_METHODS:
            raise ValueError(
                f"corr_method must be one of {sorted(ALLOWED_CORR_METHODS)}. "
                f"Got: {self.corr_method}"
            )

        if self.aggregate_mode not in ALLOWED_AGGREGATE_MODES:
            raise ValueError(
                f"aggregate_mode must be one of {sorted(ALLOWED_AGGREGATE_MODES)}. "
                f"Got: {self.aggregate_mode}"
            )

    def _validate_policies(self) -> None:
        if self.min_measurements_per_block < 1:
            raise ValueError("min_measurements_per_block must be >= 1.")

        if self.expected_lasers_count < 1:
            raise ValueError("expected_lasers_count must be >= 1.")

        if self.min_valid_lasers_per_measurement < 1:
            raise ValueError("min_valid_lasers_per_measurement must be >= 1.")

        if self.min_valid_lasers_per_measurement > self.expected_lasers_count:
            raise ValueError(
                "min_valid_lasers_per_measurement cannot exceed expected_lasers_count."
            )

        if not self.valid_status_values:
            raise ValueError("valid_status_values cannot be empty.")

        if not self.valid_boolean_values:
            raise ValueError("valid_boolean_values cannot be empty.")

    def _validate_structure(self) -> None:
        # Strong guardrails against reactivating old structural contracts.
        forbidden_structural_tokens = {
            "turno",
            "shift",
            "jornada",
        }

        text_fields = [
            self.output_dir_name,
            self.notes,
            self.block_name,
        ]
        lowered = " ".join(text_fields).lower()

        # We DO allow mentioning these terms in free notes, but not as
        # structural directory names in the config contract.
        if any(tok in self.output_dir_name.lower() for tok in forbidden_structural_tokens):
            raise ValueError(
                "output_dir_name must not encode old structural axes "
                "(turno / shift / jornada)."
            )

        # Structural sanity
        if self.require_all_6_lasers and self.expected_lasers_count != 6:
            raise ValueError(
                "require_all_6_lasers=True is only consistent when expected_lasers_count=6."
            )

        # harmless use of `lowered` to avoid lint complaints in strict envs
        _ = lowered

    def _validate_enabled_stages(self) -> None:
        if not any(
            (
                self.compute_profiles_by_measurement,
                self.compute_summary_by_date_lab,
                self.compute_intra_block_correlation,
                self.compute_profile_coherence,
                self.compute_pairwise_dates_by_lab,
                self.compute_pairwise_labs_by_date,
            )
        ):
            raise ValueError("At least one analytical stage must be enabled.")

        if self.compute_profile_coherence and not self.compute_profiles_by_measurement:
            raise ValueError(
                "compute_profile_coherence=True requires "
                "compute_profiles_by_measurement=True."
            )

    def _validate_artifacts(self) -> None:
        values = [
            self.artifacts.input_resultados_luces_csv,
            self.artifacts.input_quality_scores_by_file_csv,
            self.artifacts.base_csv,
            self.artifacts.profiles_by_measurement_csv,
            self.artifacts.summary_by_date_lab_csv,
            self.artifacts.correlation_by_date_lab_csv,
            self.artifacts.pairwise_dates_by_lab_csv,
            self.artifacts.pairwise_labs_by_date_csv,
            self.artifacts.run_metadata_json,
            self.artifacts.reports_root_dir,
            self.artifacts.global_dir,
            self.artifacts.per_date_lab_dir,
            self.artifacts.per_laser_dir,
            self.artifacts.per_lab_dir,
        ]

        empty = [v for v in values if not str(v).strip()]
        if empty:
            raise ValueError("Artifact names cannot be empty.")

        duplicates = _find_duplicates(values)
        if duplicates:
            raise ValueError(f"Artifact names contain duplicates: {sorted(duplicates)}")

    def _validate_input_schema(self) -> None:
        required = self.input_schema.required_columns()
        if len(set(required)) != len(required):
            raise ValueError(
                "Input schema required columns contain duplicates. "
                f"Required: {required}"
            )

        if self.input_schema.laser_col in {
            self.input_schema.fecha_col,
            self.input_schema.lab_col,
            self.input_schema.mid_col,
        }:
            raise ValueError("laser_col must be distinct from fecha/lab/mid columns.")

    # ======================================================
    # Helpers
    # ======================================================
    def summary(self) -> Dict[str, object]:
        """
        Compact summary for logging / orchestration.
        """
        return {
            "block_name": self.block_name,
            "version": self.version,
            "metrics": list(self.metrics),
            "corr_method": self.corr_method,
            "aggregate_mode": self.aggregate_mode,
            "expected_lasers_count": self.expected_lasers_count,
            "require_all_6_lasers": self.require_all_6_lasers,
            "drop_invalid_rows": self.drop_invalid_rows,
            "min_measurements_per_block": self.min_measurements_per_block,
            "min_valid_lasers_per_measurement": self.min_valid_lasers_per_measurement,
            "compute_profiles_by_measurement": self.compute_profiles_by_measurement,
            "compute_summary_by_date_lab": self.compute_summary_by_date_lab,
            "compute_intra_block_correlation": self.compute_intra_block_correlation,
            "compute_profile_coherence": self.compute_profile_coherence,
            "compute_pairwise_dates_by_lab": self.compute_pairwise_dates_by_lab,
            "compute_pairwise_labs_by_date": self.compute_pairwise_labs_by_date,
            "compute_anisotropy_indices": self.compute_anisotropy_indices,
            "compute_profile_distances": self.compute_profile_distances,
            "compute_pairwise_profile_correlations": self.compute_pairwise_profile_correlations,
            "output_dir_name": self.output_dir_name,
            "write_metadata_json": self.write_metadata_json,
            "notes": self.notes,
        }

    def to_dict(self) -> Dict[str, object]:
        """
        Full config as a JSON-serializable dictionary.
        """
        return asdict(self)

    def to_json_str(self, indent: int = 2) -> str:
        """
        Config serialized as JSON string.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write_json(self, path: Path) -> Path:
        """
        Write config to JSON file.

        Parameters
        ----------
        path:
            Destination JSON path.

        Returns
        -------
        Path
            The resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_str(indent=2), encoding="utf-8")
        return path

    # ======================================================
    # Factory constructors
    # ======================================================
    @classmethod
    def blind_default(cls) -> "CorrelationLasersConfig":
        """
        Recommended default for the current project.

        Blind-native:
        - no Ctrl/Exp semantics
        - Quality-derived inputs
        - 6-laser structure required
        """
        cfg = cls()
        cfg.validate()
        return cfg

    @classmethod
    def from_metrics(
        cls,
        metrics: Iterable[str],
        corr_method: str = "pearson",
        aggregate_mode: str = "mean",
        *,
        require_all_6_lasers: bool = True,
        min_measurements_per_block: int = 3,
        min_valid_lasers_per_measurement: int = 4,
    ) -> "CorrelationLasersConfig":
        """
        Construct a config with explicit metrics and core policies.
        """
        cfg = cls(
            metrics=tuple(metrics),
            corr_method=corr_method,
            aggregate_mode=aggregate_mode,
            require_all_6_lasers=require_all_6_lasers,
            min_measurements_per_block=min_measurements_per_block,
            min_valid_lasers_per_measurement=min_valid_lasers_per_measurement,
        )
        cfg.validate()
        return cfg


# ----------------------------------------------------------
# Small utility
# ----------------------------------------------------------
def _find_duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    dup: set[str] = set()
    for v in values:
        if v in seen:
            dup.add(v)
        else:
            seen.add(v)
    return dup