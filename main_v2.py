# main_v2.py
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
# ==========================================================
# Ensure module path is visible to Python
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
MODULES_DIR = BASE_DIR / "Modulos"

if not MODULES_DIR.exists():
    raise FileNotFoundError(
        f"Modules directory not found: {MODULES_DIR}"
    )

# Must be inserted BEFORE importing local modules
sys.path.insert(0, str(MODULES_DIR))

# ==========================================================
# EXP SEPTUP (from Modulos Python)
# ==========================================================
from config import ExperimentConfig
from experiment_setup import build_experiment_config, print_config
from io_dataset import DatasetIO
from normalizer import Normalizer
from manifest import ManifestWriter
from summary_table import load_manifest, summarize, print_summary




# ==========================================================
# Analysis-Ready Data Preparation (from Modulos Python)
# ==========================================================

from analysis_ready_prep import AnalysisCatalogBuilder, CSVDialectInspector
from column_role_detection import ColumnRoleDetector, ColumnRoleSpec
from measurement_time_bounds import MeasurementTimeBounds
from analysis_table_builder import AnalysisTableBuilder, TableBuildPolicy
from analysis_ready_schema_table import build_analysis_ready_schema_tables

# ==========================================================
# QUALITY LASERS  (from Modulos Python)
# ==========================================================

from quality_config import QualityConfig
from quality_metrics import calcular_metricas_luz, validar_valores_luz
from quality_runner import QualityRunner
from quality_gate import QualityGate, GatePolicy
from quality_compare import QualityCompareRunner

# ==========================================================
# QUALITY LASERS COMPARE  (from Modulos Python)
# ==========================================================
from stability_config import StabilityConfig
from stability_compare import StabilityCompare
from stability_runner import StabilityRunner

# ==========================================================
# INFORMATIONAL ANALYSIS  (from Modulos Python)
# ==========================================================

from informational_config import (
    InformationalConfig,
    InformationalConfigError,
)

from informational_bins import (
    build_bins_spec_from_config,
    BinsPolicy,
    InformationalBinsError,
)

from bins_health_report import (
    evaluate_bins_health,
    print_bins_health_report,
    BinsHealthPolicy,
    BinsHealthError,
)


from informational_states import (
    build_states_from_bins,
    StatesPolicy,
    InformationalStatesError,
)

from state_forma import (
    run_state_forma,
    FormaPolicy,
    FormaStateError,
)

from state_movimiento import (
    run_state_movimiento,
    MovimientoPolicy,
    MovimientoStateError,
)

from state_energia import (
    run_state_energia_from_level2_config,
    EnergiaPolicy,
    EnergiaStateError,
)

from state_fourier import (
    run_state_fourier_from_level2_config,
    FourierPolicy,
    FourierStateError,
)

from state_coupling_joint import (
    run_state_coupling_joint_from_level2_config,
    CouplingJointPolicy,
    CouplingJointError,
)

from state_coupling_maps import (
    run_state_coupling_maps_from_level2_config,
    CouplingMapsPolicy,
    CouplingMapsError,
)

# ==========================================================
# INFERENCE (EDIT HERE)
# ==========================================================



from change_blind_sequential import (
    run_change_blind_sequential_from_level2_config,
    BlindSequentialPolicy,
)

# ==========================================================
# PRELIMINARY EXPERIMENT PARAMETERS (EDIT HERE)
# ==========================================================
SOURCE_ROOT = BASE_DIR
TARGET_ROOT = BASE_DIR / "Mar26_clean"

DATES = [ "02Mar26", "03Mar26", "04Mar26"]
DEFAULT_SHIFT = "M"


# Toggle experiment mode
DOUBLE_BLIND = True   # True = double-blind, False = declared


# ==========================================================
# DECLARED MODE PARAMETERS (used only if DOUBLE_BLIND=False)
# ==========================================================
DATE_TO_GROUP_DECLARED = {
    # "08Ene25": "Ctrl",
    # "09Ene25": "Ctrl",
    # "10Ene25": "Exp",
}

RENAME_MAP_DECLARED = {
    # "1medcolor.csv": "1medA.csv",
}

DECLARED_LABEL_MAP = {
    # "1medA.csv": "LB",
    # "3medA.csv": "MEI",
}

# Optional late unblinding (external files)
LABELS_CSV_EXTERNAL = None  # BASE_DIR / "labels.csv"
GROUPS_CSV_EXTERNAL = None  # BASE_DIR / "groups.csv"


# ==========================================================
# MAIN (no experimental logic here)
# ==========================================================
def main() -> None:
    # ==========================================================
    # Build and validate config
    # ==========================================================
    try:
        cfg = build_experiment_config(
            source_root=SOURCE_ROOT,
            target_root=TARGET_ROOT,
            dates=DATES,
            default_shift=DEFAULT_SHIFT,
            double_blind=DOUBLE_BLIND,
            date_to_group_declared=DATE_TO_GROUP_DECLARED,
            rename_map_declared=RENAME_MAP_DECLARED,
            declared_label_map=DECLARED_LABEL_MAP,
            labels_csv_external=LABELS_CSV_EXTERNAL,
            groups_csv_external=GROUPS_CSV_EXTERNAL,
        )
    except Exception as e:
        print("[MAIN][ERROR] Failed to build/validate config.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return
    

    # --- MUST: keep stability_cfg always defined (blind-safe default) ---
    stability_cfg = cfg  # default: blind-mode uses ExperimentConfig

    try:
        print("[MAIN] Building StabilityConfig (Level 1C)...")

        # Declared-mode: builds Ctrl1/Ctrl2/Exp if mapping exists
        stability_cfg = StabilityConfig.from_experiment(
            fechas=cfg.dates,
            jornada_por_fecha=cfg.date_to_group,
        )

        stability_path = (
            cfg.target_root
            / "Reports"
            / "Level1_Stability"
            / "stability_config.json"
        )
        stability_cfg.write_json(stability_path)

        print("[MAIN][OK] StabilityConfig written to:", stability_path)

    except Exception as e:
        # Non-blocking: expected in double-blind mode
        print("[MAIN][WARN] StabilityConfig step skipped (non-blocking).")
        print(f"[MAIN][WARN] {type(e).__name__}: {e}")

    print_config(cfg)
    print("[MAIN][OK] Config built and validated.")
    
    


    # ==========================================================
    # Dataset discovery (raw files) -- uses io_dataset.py
    # ==========================================================
    try:
        io = DatasetIO(cfg, raw_subdir="Raw Data", include_lab_root_csv=True)
        items = io.collect_all(strict=False)
    except Exception as e:
        print("[MAIN][ERROR] Dataset discovery failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    if not items:
        print("[MAIN][WARN] No raw files found. Aborting.")
        return

    print(f"[MAIN] Discovered raw CSV files: {len(items)}")

    # ==========================================================
    # Normalization planning (DECLARED or DOUBLE-BLIND safe)
    # ==========================================================
    try:
        normalizer = Normalizer(cfg)
        plan = normalizer.plan(items)
    except Exception as e:
        print("[MAIN][ERROR] Normalization planning failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    print(f"[MAIN][OK] Normalization plan generated: {len(plan)} items")

    # ------------------------------------------------------
    # 3.5) Execute physical copy (materialize TARGET_ROOT)
    # ------------------------------------------------------
    try:
        print("[MAIN] Executing physical copy to TARGET_ROOT...")
        io.execute_plan(
            plan,
            dry_run=False,     # ← pon True si solo quieres simular
            overwrite=True
        )
        print("[MAIN][OK] Physical copy completed.")
    except Exception as e:
        print("[MAIN][ERROR] Physical copy failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    # Optional: show first few planned actions
    for ni in plan[:5]:
        # Normalizer v2 fields: mid/group/label
        print(f"  - {ni.src_path} -> {ni.dst_path} | mid={ni.mid} | group={ni.group} | label={ni.label}")

    
    # ------------------------------------------------------
    # 4) Manifest generation (formal record)
    # ------------------------------------------------------
    try:
        manifest_path = cfg.target_root / "manifest_all.csv"
        ManifestWriter(cfg).write(plan, manifest_path)
    except Exception as e:
        print("[MAIN][ERROR] Manifest generation failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    print(f"[MAIN][OK] Manifest written to: {manifest_path}")


    # ------------------------------------------------------
    # 5) Structural summary (human-readable, no inference)
    # ------------------------------------------------------
    try:
        manifest_path = cfg.target_root / "manifest_all.csv"
        rows = load_manifest(manifest_path)
        summary = summarize(rows)

        print("[MAIN][OK] Structural summary from manifest:")
        print_summary(summary)

    except Exception as e:
        print("[MAIN][WARN] Failed to generate summary table.")
        print(f"[MAIN][WARN] {type(e).__name__}: {e}")


    # ------------------------------------------------------
    # 6) Analysis-ready catalog + optional CSV dialect inspection
    # ------------------------------------------------------
    try:
        print("[MAIN] Building analysis catalog...")
        builder = AnalysisCatalogBuilder(cfg)
        catalog_df = builder.build(resolve_mode="compat", write=True)
        catalog_path = cfg.target_root / "analysis_catalog.csv"
        print(f"[MAIN][OK] analysis_catalog.csv written to: {catalog_path}")
        print(f"[MAIN][OK] Catalog rows: {len(catalog_df)}")

        print("[MAIN] Running CSV dialect inspector (shallow)...")
        inspector = CSVDialectInspector(cfg)
        catalog_df2, report = inspector.run(catalog_df, inspect_rows=200, write=True)
        report_path = cfg.target_root / "dialect_report.json"
        print(f"[MAIN][OK] dialect_report.json written to: {report_path}")

        # Optional quick diagnostic counts
        ok_count = int(catalog_df2["csv_dialect_ok"].sum()) if "csv_dialect_ok" in catalog_df2.columns else 0
        exists_count = int(catalog_df2["raw_exists"].sum()) if "raw_exists" in catalog_df2.columns else 0
        print(f"[MAIN][OK] raw_exists: {exists_count}/{len(catalog_df2)}")
        print(f"[MAIN][OK] csv_dialect_ok: {ok_count}/{len(catalog_df2)}")

    except Exception as e:
        print("[MAIN][ERROR] Analysis-ready preparation failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    # ------------------------------------------------------
    # 7) Column role detection (no data rewrite, no inference)
    # ------------------------------------------------------
    try:
        print("[MAIN] Detecting column roles (t_sys, t_rel, channels)...")

        role_spec = ColumnRoleSpec(
            time_system_candidates=[
                "time", "system_time", "t_sys", "timestamp"
            ],
            time_relative_candidates=[
                "t", "time_rel", "relative_time", "t_rel"
            ],
            channel_name_pool=[
                "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8"
            ],
            detect_rows=80,
            min_parse_ratio_time=0.80,
            min_parse_ratio_numeric=0.80,
            require_two_time_cols=True,
        )

        detector = ColumnRoleDetector(cfg)
        artifacts = detector.run(
            catalog_df2,
            role_spec,
            write=True
        )

        print(f"[MAIN][OK] Column roles written to: {artifacts.role_map_path}")

    except Exception as e:
        print("[MAIN][ERROR] Column role detection failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return
    # ------------------------------------------------------
    # 8) Measurement time bounds extraction (read-only)
    # ------------------------------------------------------
    try:
        print("[MAIN] Extracting measurement time bounds (t_sys / t_rel)...")

        bounds_builder = MeasurementTimeBounds(cfg)
        bounds_artifacts = bounds_builder.run(
            catalog_df2,
            roles_path=artifacts.role_map_path,  # column_roles.json
            write=True
        )

        print(f"[MAIN][OK] Measurement time bounds written to: {bounds_artifacts.time_bounds_path}")

    except Exception as e:
        print("[MAIN][ERROR] Measurement time bounds extraction failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    # ==========================================================
    # 9) Build Analysis Ready tables (Parquet)
    # ==========================================================
    try:
        print("[MAIN] Building Analysis Ready tables...")

        table_builder = AnalysisTableBuilder(
            root=cfg.target_root
        )

        table_policy = TableBuildPolicy(
            sentinels=(-111.0,),
            default_delimiter=",",
            encoding_candidates=("utf-8-sig", "utf-8", "latin-1"),
            output_layout="fecha_lab_jornada",
            filename_strategy="mid",
            overwrite=True,
        )

        table_artifacts = table_builder.run(
            catalog_df2,
            column_roles_path=artifacts.role_map_path,
            policy=table_policy,
            write_actions=True,
        )

        print(f"[MAIN][OK] Analysis Ready tables written under: {table_artifacts.output_root}")
        print(f"[MAIN][OK] Table actions log: {table_artifacts.actions_path}")

        # ==========================================================
        # FAIL-FAST CHECK: Analysis Ready must have rows
        # ==========================================================

        sample_row = catalog_df2.iloc[0]

        sample_parquet = (
            table_artifacts.output_root
            / str(sample_row["fecha"])
            / str(sample_row["lab"])
            / str(sample_row["jornada"])
            / f'{str(sample_row["mid"])}.parquet'
        )

        nrows = pq.ParquetFile(sample_parquet).metadata.num_rows

        if nrows <= 0:
            raise RuntimeError(
                f"[MAIN][FATAL] Analysis Ready parquet has 0 rows: {sample_parquet}. "
                "The error is in analysis_table_builder / analysis_table_io."
            )


    except Exception as e:
        print("[MAIN][ERROR] Failed to build Analysis Ready tables.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return


    # ==========================================================
    # 10) Validate Analysis Ready schema consistency
    # ==========================================================
    try:
        print("[MAIN] Validating Analysis Ready schema consistency...")

        build_analysis_ready_schema_tables(
            catalog_df=catalog_df2,
            column_roles_path=artifacts.role_map_path,
            analysis_ready_root=cfg.target_root / "Analysis Ready",
            output_root=cfg.target_root,
        )

        print("[MAIN][OK] Analysis Ready schema tables generated.")

    except Exception as e:
        print("[MAIN][ERROR] Analysis Ready schema validation failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    # ==========================================================
    # 11) Initialize and validate Quality configuration (Level 1A)
    # ==========================================================
    try:
        print("[MAIN] Initializing Quality configuration (Level 1A)...")

        quality_cfg = QualityConfig.from_defaults()
        quality_cfg.validate()

        # Persist configuration for reproducibility and audit
        quality_config_path = (
            cfg.target_root
            / "Reports"
            / "Level1_Quality"
            / "quality_config.json"
        )

        quality_cfg.write_json(quality_config_path)

        print("[MAIN][OK] Quality configuration validated.")
        print(f"[MAIN][OK] Quality config written to: {quality_config_path}")

    except Exception as e:
        print("[MAIN][ERROR] Quality configuration validation failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    # ==========================================================
    # 12) Compute Quality Metrics (per measurement, per channel)
    # ==========================================================
    try:
        print("[MAIN] Computing quality metrics (per channel)...")

        quality_results = []

        analysis_ready_root = cfg.target_root / "Analysis Ready"

        for row in catalog_df2.itertuples(index=False):
            mid = row.mid
            fecha = row.fecha
            lab = row.lab
            jornada = row.jornada

            parquet_path = (
                analysis_ready_root
                / fecha
                / lab
                / jornada
                / f"{mid}.parquet"
            )

            if not parquet_path.exists():
                print(f"[QUALITY][WARN] Parquet not found for mid={mid}")
                continue

            df = pd.read_parquet(parquet_path)

            # Auto-detect channels from schema (ChannelPolicy)
            channel_cols = [
                c for c in df.columns
                if c.startswith(tuple(quality_cfg.channels.allowed_prefixes))
            ]

            if not channel_cols:
                print(f"[QUALITY][WARN] No valid channels detected for mid={mid}")
                continue

            for col in channel_cols:
                # Physical validation
                val = validar_valores_luz(df, col, cfg=quality_cfg)
                if not val["valido"]:
                    continue

                metrics = calcular_metricas_luz(
                    df_luz=df,
                    columna=col,
                    nombre_archivo=mid,
                    cfg=quality_cfg,
                )

                if metrics is None:
                    continue

                metrics.update(
                    {
                        "mid": mid,
                        "fecha": fecha,
                        "lab": lab,
                        "jornada": jornada,
                    }
                )

                quality_results.append(metrics)

        print(f"[MAIN][OK] Quality metrics computed: {len(quality_results)} rows")

    except Exception as e:
        print("[MAIN][ERROR] Quality metrics computation failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return

    # ==========================================================
    # 12) Run QualityRunner (Level 1A)
    # ==========================================================
    try:
        print("[MAIN] Running QualityRunner (Level 1A)...")

        quality_runner = QualityRunner(
            root=cfg.target_root,
            config=quality_cfg,
            analysis_ready_dir_name="Analysis Ready",
        )

        # Optional: schema gate from analysis_ready_schema_table
        schema_by_file_csv = cfg.target_root / "analysis_ready_schema_by_file.csv"
        if not schema_by_file_csv.exists():
            schema_by_file_csv = None

        quality_artifacts = quality_runner.run(
            catalog_df=catalog_df2,
            schema_by_file_csv=schema_by_file_csv,
            output_dir=cfg.target_root / "Reports" / "Level1_Quality",
            write=True,
            verbose=True,
        )

        print("[MAIN][OK] QualityRunner completed.")
        print(f"[MAIN][OK] resultados_luces.csv: {quality_artifacts.resultados_luces_csv}")
        print(f"[MAIN][OK] quality_scores_by_file.csv: {quality_artifacts.scores_by_file_csv}")
        print(f"[MAIN][OK] resumen_ejecutivo.txt: {quality_artifacts.resumen_ejecutivo_txt}")

    except Exception as e:
        print("[MAIN][ERROR] QualityRunner failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return
    
    # ==========================================================
    # 13) Quality Gate — decide PASS / FAIL to Level 2
    # ==========================================================
    try:
        print("[MAIN] Running Quality Gate (Level 1A -> Level 2)...")

        gate_policy = GatePolicy(
            pass_score=quality_cfg.thresholds.pass_to_informational,
            overwrite=True,
        )

        quality_gate = QualityGate(policy=gate_policy)

        gate_artifacts = quality_gate.run(
            scores_by_file_csv=(
                cfg.target_root
                / "Reports"
                / "Level1_Quality"
                / "quality_scores_by_file.csv"
            ),
            output_dir=(
                cfg.target_root
                / "Reports"
                / "Level1_Quality"
            ),
            verbose=True,
        )

        print("[MAIN][OK] Quality Gate completed.")
        print(f"[MAIN][OK] Gate CSV: {gate_artifacts.gate_csv}")
        print(f"[MAIN][OK] PASS mids: {gate_artifacts.pass_mids_csv}")
        print(f"[MAIN][OK] FAIL mids: {gate_artifacts.fail_mids_csv}")
        print(f"[MAIN][OK] Informational queue: {gate_artifacts.informational_queue_csv}")

    except Exception as e:
        print("[MAIN][ERROR] Quality Gate failed.")
        print(f"[MAIN][ERROR] {type(e).__name__}: {e}")
        return    

    # ==========================================================
    # 14) Quality Compare (Level 1A) — visual traceability (non-blocking)
    # ==========================================================
    try:
        print("[MAIN] Running QualityCompareRunner (non-blocking)...")

        compare = QualityCompareRunner(root=cfg.target_root, config=quality_cfg)

        compare_artifacts = compare.run(
            resultados_luces_csv=(
                cfg.target_root / "Reports" / "Level1_Quality" / "resultados_luces.csv"
            ),
            schema_by_file_csv=(
                cfg.target_root / "analysis_ready_schema_by_file.csv"
            ),
            output_dir=(
                cfg.target_root / "Reports" / "Level1_Quality"
            ),
            analysis_ready_root=(
                cfg.target_root / "Analysis Ready"
            ),
            max_points=1200,
            max_files_in_plots=160,
            write=True,
            verbose=True,
        )

        print("[MAIN][OK] QualityCompareRunner completed.")
        print(f"[MAIN][OK] comparison PNG: {compare_artifacts.comparison_by_file_png}")
        if compare_artifacts.temporal_evolution_png is not None:
            print(f"[MAIN][OK] evolution PNG:   {compare_artifacts.temporal_evolution_png}")
        print(f"[MAIN][OK] per-laser reports: {len(compare_artifacts.per_laser_reports_png)}")

    except Exception as e:
        # Non-blocking by design
        print("[MAIN][WARN] QualityCompareRunner failed (non-blocking).")
        print(f"[MAIN][WARN] {type(e).__name__}: {e}")
        
    # ==========================================================
    # 15) Stability Compare (Level 1C) — Quality conservation (non-blocking)
    # ==========================================================
    try:
        print("[MAIN] Running StabilityCompare (Level 1C) (non-blocking)...")

        stability_dir = cfg.target_root / "Reports" / "Level1_Stability"
        stability_dir.mkdir(parents=True, exist_ok=True)

        # NOTE:
        # - declared-mode: stability_cfg is a StabilityConfig
        # - blind-mode:    stability_cfg is ExperimentConfig (cfg)
        sc = StabilityCompare(root=cfg.target_root, config=stability_cfg)

        schema_csv = cfg.target_root / "analysis_ready_schema_by_file.csv"
        schema_csv = schema_csv if schema_csv.exists() else None

        artifacts = sc.run(
            resultados_luces_csv=(cfg.target_root / "Reports" / "Level1_Quality" / "resultados_luces.csv"),
            scores_by_file_csv=(cfg.target_root / "Reports" / "Level1_Quality" / "quality_scores_by_file.csv"),
            schema_by_file_csv=schema_csv,
            output_dir=stability_dir,
            write=True,
            make_plots=True,
            verbose=True,
        )

        print("[MAIN][OK] StabilityCompare completed.")
        print(f"[MAIN][OK] by_laser_csv: {artifacts.by_laser_csv}")
        print(f"[MAIN][OK] summary_csv:  {artifacts.summary_csv}")
        print(f"[MAIN][OK] by_file_csv:  {artifacts.by_file_csv}")
        print(f"[MAIN][OK] report_txt:   {artifacts.report_txt}")
        if getattr(artifacts, "overview_png", None) is not None:
            print(f"[MAIN][OK] overview_png: {artifacts.overview_png}")

    except Exception as e:
        print("[MAIN][WARN] StabilityCompare skipped (non-blocking).")
        print(f"[MAIN][WARN] {type(e).__name__}: {e}")
  
    # ==========================================================
    # Level 1C — StabilityRunner
    # ==========================================================
    try:
        print("[MAIN] Running StabilityRunner (Level 1C)...")

        stability_out_dir = (
            cfg.target_root
            / "Reports"
            / "Level1_Stability"
        )

        stability_runner = StabilityRunner(
            cfg=stability_cfg,  # puede ser StabilityConfig (declared) o ExperimentConfig (blind)
            resultados_luces_csv=(
                cfg.target_root
                / "Reports"
                / "Level1_Quality"
                / "resultados_luces.csv"
            ),
            out_dir=stability_out_dir,
            enable_clustering=False,   # recomendado por ahora (low-resource)
            n_clusters=3,
            fail_quality_threshold=5.0,
        )

        stability_outputs = stability_runner.run()

        print("[MAIN][OK] StabilityRunner completed.")
        print(f"[MAIN][OK] estabilidad_detallada.csv: {stability_outputs.estabilidad_detallada_csv}")
        print(f"[MAIN][OK] perfiles_por_grupo.csv: {stability_outputs.perfiles_por_grupo_csv}")
        print(f"[MAIN][OK] comparaciones_fechas.csv: {stability_outputs.comparaciones_fechas_csv}")

        print(f"[MAIN][OK] comparativa_ctrl_exp.csv: {stability_outputs.comparativa_ctrl_exp_csv}")
        print(f"[MAIN][OK] stability_gate.csv: {stability_outputs.stability_gate_csv}")
        print(f"[MAIN][OK] informational_queue.csv: {stability_outputs.informational_queue_csv}")
        print(f"[MAIN][OK] resumen_ejecutivo: {stability_outputs.resumen_ejecutivo_txt}")

    except Exception as e:
        print("[MAIN][WARN] StabilityRunner step skipped (non-blocking).")
        print(f"[MAIN][WARN] {type(e).__name__}: {e}")

    # ======================================================================
    # LEVEL 2 — INFORMATIONAL PREFLIGHT & CONTRACT
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level2] Initializing InformationalConfig (preflight + contract)")
    print("=" * 72)

    level2_reports_dir = (
        cfg.target_root
        / "Reports"
        / "Level2_Informational"
    )

    quality_queue_path = (
        cfg.target_root
        / "Reports"
        / "Level1_Quality"
        / "informational_queue.csv"
    )

    stability_queue_path = (
        cfg.target_root
        / "Reports"
        / "Level1_Stability"
        / "informational_queue.csv"
    )

    try:
        info_cfg = InformationalConfig.from_experiment(
            cfg=cfg,
            quality_queue_path=quality_queue_path,
            stability_queue_path=stability_queue_path,
            level2_folder_name="Level2_Informational",
            metric_primary="JS",
            alpha=0.05,
            correction="none",
            coupling_lags_samples=(0, 1),
        )

        preflight_report = info_cfg.preflight(
            strict=True,
            parquet_sample_n=25,
            verbose=True,
        )

        level2_reports_dir.mkdir(parents=True, exist_ok=True)
        level2_config_path = level2_reports_dir / "level2_config.json"
        info_cfg.write_json(level2_config_path)

        print("[Level2][OK] InformationalConfig ready.")
        print(f"[Level2][OK] Contract written to: {level2_config_path}")

    except InformationalConfigError as e:
        print("\n" + "!" * 72)
        print("[Level2][FAIL] Informational preflight failed.")
        print(str(e))
        print("Fix the errors above before running informational analysis.")
        print("!" * 72)
        return

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level2][FAIL] Unexpected error during InformationalConfig setup.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return

    # ======================================================================
    # LEVEL 2 — INFORMATIONAL BINS (frozen bins contract)
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level2] Building informational bins (data-driven partitions)")
    print("=" * 72)

    try:
        # Policy can be tuned later; defaults are data-driven and safe
        bins_policy = BinsPolicy()

        bins_artifacts = build_bins_spec_from_config(
            level2_cfg=info_cfg,          # ← InformationalConfig instance
            policy=bins_policy,
            parquet_engine="auto",        # let pandas decide (pyarrow / fastparquet)
        )

        print("[Level2][OK] Informational bins built successfully.")
        print(f"[Level2][OK] bins_spec.json  → {bins_artifacts.bins_spec_json}")
        print(f"[Level2][OK] bins_summary.csv → {bins_artifacts.bins_summary_csv}")

    except InformationalBinsError as e:
        print("\n" + "!" * 72)
        print("[Level2][FAIL] Informational bins construction failed.")
        print(str(e))
        print("Fix the issues above before proceeding to informational states.")
        print("!" * 72)
        return  # HARD STOP: states/inference depend on frozen bins

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level2][FAIL] Unexpected error during informational bins setup.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return

    # ======================================================================
    # LEVEL 2 — BINS HEALTH (global validation + clipping scan)
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level2] Evaluating bins health (global diagnostics)")
    print("=" * 72)

    try:
        health_policy = BinsHealthPolicy(
            scan_parquets=True,
            parquet_sample_n=30,
            per_parquet_row_cap=200_000,
            clip_warn=0.005,
            clip_fail=0.02,
            nan_warn=0.001,
            nan_fail=0.01,
            seed=123456,
        )

        health_report = evaluate_bins_health(
            bins_spec_json=level2_reports_dir / "bins_spec.json",
            informational_queue_csv=quality_queue_path,   # uses parquet_path to sample
            out_dir=level2_reports_dir,
            policy=health_policy,
        )

        print_bins_health_report(health_report)

        if health_report.get("status") == "FAIL":
            print("\n" + "!" * 72)
            print("[Level2][FAIL] Bins health indicates critical issues.")
            print("Do not proceed to informational states until fixed.")
            print("!" * 72)
            return  # HARD STOP

    except BinsHealthError as e:
        print("\n" + "!" * 72)
        print("[Level2][FAIL] Bins health evaluation failed.")
        print(str(e))
        print("!" * 72)
        return

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level2][FAIL] Unexpected error during bins health evaluation.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return

    # ======================================================================
    # LEVEL 2 — INFORMATIONAL STATES (PMFs + Entropies)
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level2] Building informational states (PMFs, entropies, coupling)")
    print("=" * 72)

    try:
        states_policy = StatesPolicy(
            log_base=2.0,
            remove_dc=True,
            prefer_column_read=True,
            pmf_batch_rows=250_000,
            require_spectral=False,      # ← cambia a True si quieres forzarlo
            coupling_required=False,     # ← cambia a True si coupling es obligatorio
        )

        states_artifacts = build_states_from_bins(
            bins_spec_json=level2_reports_dir / "bins_spec.json",
            quality_queue_csv=quality_queue_path,
            stability_queue_csv=stability_queue_path,
            output_dir=level2_reports_dir,
            policy=states_policy,
        )

        print("[Level2][OK] Informational states built successfully.")
        print(f"[Level2][OK] States summary: {states_artifacts.states_summary_csv}")
        print(f"[Level2][OK] PMFs stored at: {states_artifacts.pmf_long_path}")
        print(f"[Level2][OK] Coupling MI: {states_artifacts.coupling_mi_csv}")

    except InformationalStatesError as e:
        print("\n" + "!" * 72)
        print("[Level2][FAIL] Informational states construction failed.")
        print(str(e))
        print("Fix the error above before proceeding to inference.")
        print("!" * 72)
        return  # HARD STOP

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level2][FAIL] Unexpected error during informational states.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return
    
    # ======================================================================
    # LEVEL 3 — STATE FORMA (Amplitude) — Presentation / Organization
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level3] Building FORMA state (Amplitude)")
    print("=" * 72)

    try:
        forma_policy = FormaPolicy(
            split_by_jkey=True,          # mantiene tus salidas organizadas por fecha/lab/etc
            write_pmf_long=False,        # True si quieres forma_pmf_long.csv
            redact_labels_if_blind=True, # seguridad doble ciego
        )

        forma_artifacts = run_state_forma(
            level2_reports_dir=level2_reports_dir,
            out_dir=(
                cfg.target_root
                / "Reports"
                / "Level3_Informational"
                / "Forma"
            ),
            policy=forma_policy,
            verbose=True,
        )

        print("[Level3][OK] FORMA state built successfully.")
        print(f"[Level3][OK] Summary: {forma_artifacts.forma_summary_csv}")

        if forma_artifacts.by_jkey_dir is not None:
            print(f"[Level3][OK] by_jkey outputs: {forma_artifacts.by_jkey_dir}")

    except FormaStateError as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] FORMA state failed.")
        print(str(e))
        print("Fix the error above before proceeding to other states.")
        print("!" * 72)
        return  # HARD STOP

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] Unexpected error during FORMA state.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return

    # ======================================================================
    # LEVEL 3 — STATE MOVIMIENTO (Increments) — Presentation / Organization
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level3] Building MOVIMIENTO state (Increments)")
    print("=" * 72)

    try:
        movimiento_policy = MovimientoPolicy(
            split_by_jkey=True,          # mantiene organización por jornada/lab/etc
            write_pmf_long=False,        # True solo si quieres auditoría detallada
            redact_labels_if_blind=True, # seguridad doble ciego
        )

        movimiento_artifacts = run_state_movimiento(
            level2_reports_dir=level2_reports_dir,
            out_dir=(
                cfg.target_root
                / "Reports"
                / "Level3_Informational"
                / "Movimiento"
            ),
            policy=movimiento_policy,
            verbose=True,
        )

        print("[Level3][OK] MOVIMIENTO state built successfully.")
        print(f"[Level3][OK] Summary: {movimiento_artifacts.movimiento_summary_csv}")

        if movimiento_artifacts.by_jkey_dir is not None:
            print(f"[Level3][OK] by_jkey outputs: {movimiento_artifacts.by_jkey_dir}")

    except MovimientoStateError as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] MOVIMIENTO state failed.")
        print(str(e))
        print("Fix the error above before proceeding to other states.")
        print("!" * 72)
        return  # HARD STOP

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] Unexpected error during MOVIMIENTO state.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return

    # ======================================================================
    # LEVEL 3 — STATE: ENERGIA (Spectral Energy Entropy)
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level3] Building ENERGIA state (spectral energy entropy)")
    print("=" * 72)

    level3_energia_dir = (
        cfg.target_root
        / "Reports"
        / "Level3_Informational"
        / "Energia"
    )

    try:
        energia_policy = EnergiaPolicy(
            split_by_jkey=True,
            redact_labels_if_blind=True,
            write_pmf_long=False,       # 🔒 default OFF (blind-safe)
            write_bands_long=False,     # diagnostics only
            write_psd_cache=True,       # ✅ IMPORTANT: enables Fourier reuse
            block_selection="random",
            detrend_mean=True,
            min_blocks_required=1,
            pseudocount=0.0,
        )

        energia_artifacts = run_state_energia_from_level2_config(
            level2_config_json=level2_config_path,
            out_dir=level3_energia_dir,
            policy=energia_policy,
            verbose=True,
        )

        print("[Level3][OK] ENERGIA state completed.")
        print(f"[Level3][OK] Summary: {energia_artifacts.energia_summary_csv}")

    except EnergiaStateError as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] ENERGIA state failed (contract error).")
        print(str(e))
        print("Fix the issues above before proceeding.")
        print("!" * 72)
        return  # HARD STOP

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] Unexpected error during ENERGIA state.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return

    # ======================================================================
    # LEVEL 3 — FOURIER STATE (Modal spectral entropy, PSD-cache-first)
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level3] Running FOURIER state (spectral informational)")
    print("=" * 72)

    level3_fourier_dir = (
        cfg.target_root
        / "Reports"
        / "Level3_Informational"
        / "Fourier"
    )

    try:
        fourier_policy = FourierPolicy(
            # Organization
            split_by_jkey=True,

            # Blind / declared safety
            redact_labels_if_blind=True,
            drop_archivo=True,
            drop_parquet_path=True,

            # PSD-cache-first (STRICT)
            psd_first_required=True,
            allow_raw_fallback=False,

            # Spectral window
            fmin=0.0,
            fmax=None,  # use bins_spec spectral edges per laser

            # PMF / entropy
            alpha=0.0,
            exclude_dc=True,

            # Outputs
            write_pmf_long=False,   # ⚠️ heavy, keep False unless debugging
        )

        fourier_artifacts = run_state_fourier_from_level2_config(
            level2_config_json=level2_config_path,
            out_dir=level3_fourier_dir,
            policy=fourier_policy,
            verbose=True,
        )

        print("[Level3][OK] FOURIER state completed.")
        print(f"[Level3][OK] Summary: {fourier_artifacts.fourier_summary_csv}")
        print(f"[Level3][OK] Meta:    {fourier_artifacts.run_meta_json}")

    except FourierStateError as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] FOURIER state failed.")
        print(str(e))
        print("Fix the error above before continuing.")
        print("!" * 72)
        return  # HARD STOP

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] Unexpected error during FOURIER state.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return


    # ======================================================================
    # LEVEL 3 — STATE: COUPLING JOINT (Joint distributions only)
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level3] Building COUPLING JOINT state (joint distributions)")
    print("=" * 72)

    level3_coupling_joint_dir = (
        cfg.target_root
        / "Reports"
        / "Level3_Informational"
        / "Acoplamiento"
        / "Joint"
    )

    try:
        coupling_joint_policy = CouplingJointPolicy(
            # Output
            write_counts=True,
            prefer_parquet=True,          # CSV fallback automático si no hay pyarrow
            split_by_jkey=True,

            # Data quality
            min_Nij_per_pair=1,           # seguro para comenzar
            validate_edges=True,

            # Blind / declared
            analysis_mode="blind" if DOUBLE_BLIND else "declared",
            blind_level="soft",           # "strong" si luego quieres ocultar fecha/lab/etc
            jkey_strategy="session",      # coherente con el resto del pipeline

            # Performance
            read_columns_only=True,
            max_rows_buffer=200_000,

            # Logging
            verbose=True,
        )

        coupling_joint_artifacts = run_state_coupling_joint_from_level2_config(
            level2_config_json=level2_config_path,
            out_dir=level3_coupling_joint_dir,
            policy=coupling_joint_policy,
        )

        print("[Level3][OK] COUPLING JOINT state completed.")
        print(f"[Level3][OK] Summary: {coupling_joint_artifacts.joint_summary_csv}")

        if coupling_joint_artifacts.joint_counts_long_path is not None:
            print(
                "[Level3][OK] Joint counts:",
                coupling_joint_artifacts.joint_counts_long_path,
            )

        print(
            "[Level3][OK] Run meta:",
            coupling_joint_artifacts.joint_run_meta_json,
        )

    except CouplingJointError as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] COUPLING JOINT state failed (contract error).")
        print(str(e))
        print("Fix the issues above before proceeding to coupling maps.")
        print("!" * 72)
        return  # HARD STOP

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] Unexpected error during COUPLING JOINT state.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return

    # ======================================================================
    # LEVEL 3 — COUPLING MAPS (MI / Hcond / Urel)
    # ======================================================================

    print("\n" + "=" * 72)
    print("[Level3] Building COUPLING MAPS (MI / Hcond / Urel)")
    print("=" * 72)

    level3_coupling_maps_dir = (
        cfg.target_root
        / "Reports"
        / "Level3_Informational"
        / "Acoplamiento"
        / "Maps"
    )

    try:
        maps_policy = CouplingMapsPolicy(
            # Core math
            pseudocount=0.0,  # keep 0.0 unless you *really* want dense smoothing

            # Organization
            split_by_jkey=True,
            write_by_jkey_json=True,

            # Outputs (PNG optional)
            export_png=True,
            export_csv_matrices=True,

            # Blind / declared
            analysis_mode=("blind" if DOUBLE_BLIND else "declared"),
            blind_level="soft",  # "soft" or "strong"

            # Performance
            stream_processing=True,
            assume_sorted_by_mid=True,
            batch_rows=250_000,

            # Housekeeping
            overwrite_outputs=True,
            verbose=True,
        )

        maps_artifacts = run_state_coupling_maps_from_level2_config(
            level2_config_json=level2_config_path,
            out_dir=level3_coupling_maps_dir,
            policy=maps_policy,
        )

        print("[Level3][OK] COUPLING MAPS built successfully.")
        print(f"[Level3][OK] Summary: {maps_artifacts.coupling_summary_csv}")
        print(f"[Level3][OK] Meta:    {maps_artifacts.run_meta_json}")

    except CouplingMapsError as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] COUPLING MAPS failed (contract or data issue).")
        print(str(e))
        print("Fix the issue above before continuing.")
        print("!" * 72)
        return  # HARD STOP

    except Exception as e:
        print("\n" + "!" * 72)
        print("[Level3][FAIL] Unexpected error during COUPLING MAPS.")
        print(f"{type(e).__name__}: {e}")
        print("!" * 72)
        return
            
    # ==========================================================
    # Level 4A — Blind Sequential Change (NO inference / NO labels)
    # ==========================================================

    print("\n" + "=" * 72)
    print("[Level4A][BlindSequential] Computing sequential informational change (blind mode)")
    print("=" * 72)

    blind_policy = BlindSequentialPolicy(
        kl_epsilon=1e-12,
        verbose=True,
    )


    blind_seq_artifacts = run_change_blind_sequential_from_level2_config(
        level2_config_json=level2_config_path,
        queue_csv=quality_queue_path,   # 👈 ESTA ES LA CORRECCIÓN
        policy=blind_policy,
    )

    print("[Level4A][BlindSequential][OK]")

    for art in blind_seq_artifacts:
        print(f"  Output directory : {art.out_dir}")
        print(f"  General table    : {art.table_general_csv}")
        print(f"  Detailed table   : {art.table_detailed_csv}")
        print(f"  Meta             : {art.meta_json}")


        
if __name__ == "__main__":
    main()
