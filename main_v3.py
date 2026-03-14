from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

# ==========================================================
# Ensure module path is visible to Python
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
MODULES_DIR = BASE_DIR / "Modulos"

if not MODULES_DIR.exists():
    raise FileNotFoundError(f"Modules directory not found: {MODULES_DIR}")

sys.path.insert(0, str(MODULES_DIR))

# ==========================================================
# EXPERIMENTAL SETUP (FROM Modulos)
# ==========================================================
from config import ExperimentConfig
from io_dataset import DatasetIO
from normalizer import Normalizer
from manifest import ManifestWriter
from summary_table import load_manifest, summarize, print_summary


# ==========================================================
# Analysis-Ready Data Preparation (from Modulos )
# ==========================================================
from analysis_ready_prep import AnalysisCatalogBuilder, CSVDialectInspector
from column_role_detection import ColumnRoleDetector, ColumnRoleSpec
from measurement_time_bounds import MeasurementTimeBounds
from analysis_table_builder import AnalysisTableBuilder, TableBuildPolicy
from analysis_ready_schema_table import build_analysis_ready_schema_tables


# ==========================================================
# QUALITY LASERS  (from Modulos )
# ==========================================================
from quality_config import QualityConfig
from quality_runner import QualityRunner
from quality_gate import QualityGate, GatePolicy
from quality_compare import QualityCompareRunner, QualityCompareConfig


# ==========================================================
# CORRELATION LASERS  (from Modulos )
# ==========================================================

from correlation_lasers_config import CorrelationLasersConfig
from correlation_lasers_runner import CorrelationLasersRunner
from correlation_lasers_compare import CorrelationLasersCompare



from informational_config import (
    InformationalConfig,
    InformationalConfigError,
)

from informational_bins import (
    BinsPolicy,
    build_bins_spec_from_config,
)

from bins_health_report import (
    BinsHealthPolicy,
    evaluate_bins_health,
    print_bins_health_report,
)

# ==========================================================
# CORRELATION LASERS PARAMETERS (EDIT HERE)
# ==========================================================
CORRELATION_LASERS_ENABLED = True

CORRELATION_LASERS_METRICS = (
    "calidad_general",
    "snr_db",
    "coef_variacion",
)

CORRELATION_LASERS_CORR_METHOD = "pearson"
CORRELATION_LASERS_AGGREGATE_MODE = "mean"

CORRELATION_LASERS_RUNNER_ENABLED = True
CORRELATION_LASERS_WRITE = True
CORRELATION_LASERS_VERBOSE = True

CORRELATION_LASERS_COMPARE_ENABLED = True
CORRELATION_LASERS_COMPARE_WRITE = True
CORRELATION_LASERS_COMPARE_VERBOSE = True



# ==========================================================
# INFORMATIONAL ANALYSIS PARAMETERS (EDIT HERE)
# ==========================================================
INFORMATIONAL_CONFIG_ENABLED = True

INFORMATIONAL_LEVEL4_FOLDER_NAME = "Level4_Informational"
INFORMATIONAL_METRIC_PRIMARY = "JS"
INFORMATIONAL_ALPHA = 0.05
INFORMATIONAL_CORRECTION = "none"
INFORMATIONAL_COUPLING_LAGS = (0, 1)

INFORMATIONAL_CONFIG_WRITE = True
INFORMATIONAL_CONFIG_VERBOSE = True

# ==========================================================
# INFORMATIONAL BINS PARAMETERS (EDIT HERE)
# ==========================================================
INFORMATIONAL_BINS_ENABLED = True
INFORMATIONAL_BINS_VERBOSE = True

# ==========================================================
# INFORMATIONAL BINS HEALTH PARAMETERS (EDIT HERE)
# ==========================================================
INFORMATIONAL_BINS_HEALTH_ENABLED = True
INFORMATIONAL_BINS_HEALTH_VERBOSE = True
INFORMATIONAL_BINS_HEALTH_SCAN_PARQUETS = True
INFORMATIONAL_BINS_HEALTH_SAMPLE_N = 30
INFORMATIONAL_BINS_HEALTH_ROW_CAP = 200_000

# ==========================================================
# USER PARAMETERS (EDIT HERE)
# ==========================================================
SOURCE_ROOT = BASE_DIR
TARGET_ROOT = BASE_DIR / "Mar26_1"

SELECTED_DATES: Optional[List[str]] = None
SELECTED_LABS: Optional[List[str]] = None

DOUBLE_BLIND = True

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

LABELS_CSV_EXTERNAL = None
GROUPS_CSV_EXTERNAL = None


# ==========================================================
# DISCOVERY RULES
# ==========================================================
DATE_PATTERN = re.compile(r"^\d{2}[A-Za-z]{3}\d{2}$")

IGNORED_ROOT_DIRS = {
    "Modulos",
    "__pycache__",
    ".git",
    ".vscode",
    "Reports",
    "Analysis Ready",
}

IGNORED_LAB_DIRS = {
    "__pycache__",
    ".git",
    ".vscode",
}


def _is_valid_dir(p: Path) -> bool:
    return p.exists() and p.is_dir()


def is_valid_date_folder(name: str) -> bool:
    """
    Accept folders like:
        02Mar26
        15Ago26
        29Ene26
    """
    return bool(DATE_PATTERN.fullmatch(name))


def is_ignored_root_dir(name: str) -> bool:
    return name in IGNORED_ROOT_DIRS or name.startswith(".")


def is_ignored_lab_dir(name: str) -> bool:
    return name in IGNORED_LAB_DIRS or name.startswith(".")


def discover_dates_and_labs(
    source_root: Path,
    selected_dates: Optional[Iterable[str]] = None,
    selected_labs: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Discover experiment structure under:

        source_root/
            FECHA/
                LAB/
                    ...

    Returns:
        (dates, labs)
    """
    if not _is_valid_dir(source_root):
        raise FileNotFoundError(
            f"source_root does not exist or is not a directory: {source_root}"
        )

    selected_dates_set = set(selected_dates) if selected_dates is not None else None
    selected_labs_set = set(selected_labs) if selected_labs is not None else None

    dates_found: List[str] = []
    labs_found_set: set[str] = set()

    for date_dir in sorted(source_root.iterdir()):
        if not date_dir.is_dir():
            continue

        date_name = date_dir.name

        if is_ignored_root_dir(date_name):
            continue

        if not is_valid_date_folder(date_name):
            continue

        if selected_dates_set is not None and date_name not in selected_dates_set:
            continue

        labs_in_date = []
        for p in sorted(date_dir.iterdir()):
            if not p.is_dir():
                continue

            lab_name = p.name

            if is_ignored_lab_dir(lab_name):
                continue

            if selected_labs_set is not None and lab_name not in selected_labs_set:
                continue

            labs_in_date.append(p)

        if not labs_in_date:
            print(f"[DISCOVERY][WARN] Date folder without labs: {date_name}")
            continue

        dates_found.append(date_name)
        labs_found_set.update(p.name for p in labs_in_date)

    if not dates_found:
        raise ValueError("No valid date -> lab structure was found in source_root.")

    labs_found = sorted(labs_found_set)

    if not labs_found:
        raise ValueError("No laboratories were detected under the selected dates.")

    return dates_found, labs_found


def build_config() -> ExperimentConfig:
    dates, labs = discover_dates_and_labs(
        source_root=SOURCE_ROOT,
        selected_dates=SELECTED_DATES,
        selected_labs=SELECTED_LABS,
    )

    print(f"[DISCOVERY] dates detected : {dates}")
    print(f"[DISCOVERY] labs detected  : {labs}")

    if DOUBLE_BLIND:
        label_mode = "blind"
        group_mode = "by_date"
        labels_csv = None
        groups_csv = None
        declared_label_map = {}
        date_to_group = {}
        rename_map = {}
    else:
        label_mode = "declared" if DECLARED_LABEL_MAP else "blind"
        group_mode = "declared_ctrl_exp" if DATE_TO_GROUP_DECLARED else "by_date"
        labels_csv = LABELS_CSV_EXTERNAL
        groups_csv = GROUPS_CSV_EXTERNAL
        declared_label_map = DECLARED_LABEL_MAP
        date_to_group = DATE_TO_GROUP_DECLARED
        rename_map = RENAME_MAP_DECLARED

    cfg = ExperimentConfig(
        source_root=SOURCE_ROOT,
        target_root=TARGET_ROOT,
        dates=dates,
        labs=labs,
        default_shift=None,
        rename_map=rename_map,
        label_mode=label_mode,
        default_label="UNK",
        declared_label_map=declared_label_map,
        labels_csv=labels_csv,
        group_mode=group_mode,
        default_group="UNK",
        date_to_group=date_to_group,
        groups_csv=groups_csv,
    )

    cfg.validate()
    return cfg


def print_config_summary(cfg: ExperimentConfig) -> None:
    s = cfg.summary()

    print("=" * 72)
    print("[MAIN_V3] Experiment configuration")
    print("=" * 72)
    print(f"source_root       : {s['source_root']}")
    print(f"target_root       : {s['target_root']}")
    print(f"dates_count       : {s['dates_count']}")
    print(f"dates             : {s['dates']}")
    print(f"labs_count        : {s['labs_count']}")
    print(f"labs              : {s['labs']}")
    print(f"default_shift     : {s['default_shift']}")
    print(f"label_mode        : {s['label_mode']}")
    print(f"group_mode        : {s['group_mode']}")
    print(f"rename_map_count  : {s['rename_map_count']}")
    print(f"labels_csv        : {s['labels_csv']}")
    print(f"groups_csv        : {s['groups_csv']}")
    print("=" * 72)


def main() -> None:
    # ------------------------------------------------------
    # 1) Build config
    # ------------------------------------------------------
    try:
        cfg = build_config()
    except Exception as e:
        print("[MAIN_V3][ERROR] Failed to build ExperimentConfig.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print_config_summary(cfg)
    print("[MAIN_V3][OK] Virgin bootstrap completed.")

    # ------------------------------------------------------
    # 2) Level 0A — raw dataset discovery only
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 0A: raw dataset discovery...")

        io = DatasetIO(
            cfg=cfg,
            raw_subdir="Raw Data",
            include_lab_root_csv=True,
        )

        items = io.collect_all(strict=False)

    except Exception as e:
        print("[MAIN_V3][ERROR] Level 0A discovery failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    if not items:
        print("[MAIN_V3][WARN] No raw files found.")
        print("[MAIN_V3][NEXT] Check whether lab folders contain 'Raw Data/' or CSV files directly.")
        return

    print(f"[MAIN_V3][OK] Raw files discovered: {len(items)}")

    # ------------------------------------------------------
    # 3) Preview first discovered items
    # ------------------------------------------------------
    preview_n = min(10, len(items))
    print(f"[MAIN_V3] Preview of first {preview_n} discovered raw items:")

    for i, item in enumerate(items[:preview_n], start=1):
        print(
            f"  [{i}] "
            f"date={item.date} | "
            f"lab={item.lab} | "
            f"src_name={item.src_name} | "
            f"src={item.src_path}"
        )

    print("[MAIN_V3][NEXT] Level 0A discovery is ready.")
    print("[MAIN_V3][NEXT] Next iteration: normalization planning (without physical copy).")

    # ------------------------------------------------------
    # 4) Level 0B — normalization planning (no copy yet)
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 0B: normalization planning...")

        normalizer = Normalizer(cfg)
        plan = normalizer.plan(items)

    except Exception as e:
        print("[MAIN_V3][ERROR] Normalization planning failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    if not plan:
        print("[MAIN_V3][WARN] Normalization plan is empty.")
        return

    print(f"[MAIN_V3][OK] Normalization plan generated: {len(plan)} items")

    preview_n = min(10, len(plan))
    print(f"[MAIN_V3] Preview of first {preview_n} normalization items:")

    for i, ni in enumerate(plan[:preview_n], start=1):
        print(
            f"  [{i}] "
            f"{ni.src_path} -> {ni.dst_path} | "
            f"mid={ni.mid} | "
            f"group={ni.group} | "
            f"label={ni.label}"
        )
    # ------------------------------------------------------
    # 5) Level 0C — manifest generation
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 0C: manifest generation...")

        manifest_path = cfg.target_root / "manifest_all.csv"
        ManifestWriter(cfg).write(plan, manifest_path)

    except Exception as e:
        print("[MAIN_V3][ERROR] Manifest generation failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print(f"[MAIN_V3][OK] Manifest written to: {manifest_path}")
    print("[MAIN_V3][NEXT] Level 0C manifest is ready.")
    print("[MAIN_V3][NEXT] Next iteration: structural summary from manifest.")

    # ------------------------------------------------------
    # 6) Level 0D — structural summary from manifest
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 0D: structural summary from manifest...")

        rows = load_manifest(manifest_path)
        summary = summarize(rows)

        print("[MAIN_V3][OK] Structural summary from manifest:")
        print_summary(summary)

    except Exception as e:
        print("[MAIN_V3][ERROR] Structural summary failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print("[MAIN_V3][NEXT] Level 0D summary is ready.")    

    # ------------------------------------------------------
    # 7) Level 1A — Analysis-Ready catalog + CSV dialect inspection
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 1A: analysis-ready catalog build...")

        builder = AnalysisCatalogBuilder(cfg)
        catalog_df = builder.build(resolve_mode="compat", write=True)

        catalog_path = cfg.target_root / "analysis_catalog.csv"

        print(f"[MAIN_V3][OK] analysis_catalog.csv written to: {catalog_path}")
        print(f"[MAIN_V3][OK] Catalog rows: {len(catalog_df)}")

    except Exception as e:
        print("[MAIN_V3][ERROR] Analysis-ready catalog build failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    try:
        print("[MAIN_V3] Running Level 1B: CSV dialect inspection...")

        inspector = CSVDialectInspector(cfg)
        catalog_df2, report = inspector.run(
            catalog_df,
            inspect_rows=200,
            write=True,
        )

        report_path = cfg.target_root / "dialect_report.json"

        print(f"[MAIN_V3][OK] dialect_report.json written to: {report_path}")

        ok_count = (
            int(catalog_df2["csv_dialect_ok"].sum())
            if "csv_dialect_ok" in catalog_df2.columns
            else 0
        )
        exists_count = (
            int(catalog_df2["raw_exists"].sum())
            if "raw_exists" in catalog_df2.columns
            else 0
        )

        print(f"[MAIN_V3][OK] raw_exists: {exists_count}/{len(catalog_df2)}")
        print(f"[MAIN_V3][OK] csv_dialect_ok: {ok_count}/{len(catalog_df2)}")

    except Exception as e:
        print("[MAIN_V3][ERROR] CSV dialect inspection failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print("[MAIN_V3][NEXT] Level 1A/1B analysis-ready catalog is ready.")    

    # ------------------------------------------------------
    # 8) Level 1C — column role detection
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 1C: column role detection...")

        role_spec = ColumnRoleSpec(
            time_system_candidates=[
                "Time",
                "Hora",
                "System_Time",
                "t_sys",
            ],
            time_relative_candidates=[
                "Time 2",
                "Tiempo_Relativo",
                "Relative_Time",
                "t_rel",
            ],
            channel_name_pool=[
                "Laser_1",
                "Laser_2",
                "Laser_3",
                "Laser_4",
                "Laser_5",
                "Laser_6",
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
            write=True,
            role_map_name="column_roles.json",
            default_delimiter=",",
        )

        print(f"[MAIN_V3][OK] column_roles.json written to: {artifacts.role_map_path}")

    except Exception as e:
        print("[MAIN_V3][ERROR] Column role detection failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print("[MAIN_V3][NEXT] Level 1C column role detection is ready.")    

    # ------------------------------------------------------
    # 9) Level 1D — measurement time bounds
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 1D: measurement time bounds...")

        bounds_builder = MeasurementTimeBounds(cfg)
        bounds_artifacts = bounds_builder.run(
            catalog_df2,
            roles_path=cfg.target_root / "column_roles.json",
            write=True,
            output_name="measurement_time_bounds.json",
            default_delimiter=",",
        )

        print(f"[MAIN_V3][OK] measurement_time_bounds.json written to: {bounds_artifacts.time_bounds_path}")

    except Exception as e:
        print("[MAIN_V3][ERROR] Measurement time bounds failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return
    print("[MAIN_V3][NEXT] Level 1D measurement time bounds is ready.")
    
    
    # ------------------------------------------------------
    # 10) Level 1E — analysis table build
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 1E: analysis table build...")

        table_builder = AnalysisTableBuilder(cfg.target_root)
        table_policy = TableBuildPolicy(
            sentinels=(-111.0,),
            default_delimiter=",",
            encoding_candidates=("utf-8-sig", "utf-8", "latin-1"),
            output_layout="fecha_lab",
            filename_strategy="mid",
            overwrite=True,
        )

        table_artifacts = table_builder.run(
            catalog_df2,
            column_roles_path=cfg.target_root / "column_roles.json",
            policy=table_policy,
            write_actions=True,
        )

        print(f"[MAIN_V3][OK] Analysis Ready root: {table_artifacts.output_root}")
        print(f"[MAIN_V3][OK] table_actions.jsonl written to: {table_artifacts.actions_path}")

    except Exception as e:
        print("[MAIN_V3][ERROR] Analysis table build failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print("[MAIN_V3][NEXT] Level 1E analysis table build is ready.")    
    
    # ------------------------------------------------------
    # 11) Level 1F — analysis-ready schema validation tables
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 1F: analysis-ready schema validation...")

        build_analysis_ready_schema_tables(
            catalog_df=catalog_df2,
            column_roles_path=cfg.target_root / "column_roles.json",
            analysis_ready_root=cfg.target_root / "Analysis Ready",
            output_root=cfg.target_root,
        )

        print(
            "[MAIN_V3][OK] analysis_ready_schema_by_file.csv written to: "
            f"{cfg.target_root / 'analysis_ready_schema_by_file.csv'}"
        )
        print(
            "[MAIN_V3][OK] analysis_ready_schema_table.csv written to: "
            f"{cfg.target_root / 'analysis_ready_schema_table.csv'}"
        )

    except Exception as e:
        print("[MAIN_V3][ERROR] Analysis-ready schema validation failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print("[MAIN_V3][NEXT] Level 1F analysis-ready schema validation is ready.")    
    
    # ------------------------------------------------------
    # 12) Level 2A — quality run
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 2A: quality run...")

        quality_cfg = QualityConfig()
        quality_cfg.validate()

        quality_runner = QualityRunner(
            root=cfg.target_root,
            config=quality_cfg,
            analysis_ready_dir_name="Analysis Ready",
            default_output_dir="Reports/Level1_Quality",
        )

        quality_artifacts = quality_runner.run(
            catalog_df=catalog_df2,
            schema_by_file_csv=cfg.target_root / "analysis_ready_schema_by_file.csv",
            write=True,
            verbose=True,
        )

        print(f"[MAIN_V3][OK] resultados_luces.csv written to: {quality_artifacts.resultados_luces_csv}")
        print(f"[MAIN_V3][OK] datos_completos_luces.csv written to: {quality_artifacts.datos_completos_luces_csv}")
        print(f"[MAIN_V3][OK] quality_scores_by_file.csv written to: {quality_artifacts.scores_by_file_csv}")
        print(f"[MAIN_V3][OK] quality_summary_by_lab.csv written to: {quality_artifacts.summary_by_lab_csv}")
        print(f"[MAIN_V3][OK] quality_summary_by_date_lab.csv written to: {quality_artifacts.summary_by_date_lab_csv}")
        print(f"[MAIN_V3][OK] quality_summary_by_laser.csv written to: {quality_artifacts.summary_by_laser_csv}")
        print(f"[MAIN_V3][OK] resumen_ejecutivo.txt written to: {quality_artifacts.resumen_ejecutivo_txt}")

    except Exception as e:
        print("[MAIN_V3][ERROR] Quality run failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print("[MAIN_V3][NEXT] Level 2A quality run is ready.")

    # ------------------------------------------------------
    # 13) Level 2B — quality gate
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 2B: quality gate...")

        gate_policy = GatePolicy(
            pass_score=quality_cfg.thresholds.pass_to_informational,
            overwrite=True,
            require_min_channels_if_available=True,
        )

        quality_gate = QualityGate(policy=gate_policy)
        gate_artifacts = quality_gate.run(
            scores_by_file_csv=quality_artifacts.scores_by_file_csv,
            output_dir=cfg.target_root / "Reports" / "Level1_Quality",
            verbose=True,
        )

        print(f"[MAIN_V3][OK] quality_gate.csv written to: {gate_artifacts.gate_csv}")
        print(f"[MAIN_V3][OK] pass_mids.csv written to: {gate_artifacts.pass_mids_csv}")
        print(f"[MAIN_V3][OK] fail_mids.csv written to: {gate_artifacts.fail_mids_csv}")
        print(f"[MAIN_V3][OK] informational_queue.csv written to: {gate_artifacts.informational_queue_csv}")

    except Exception as e:
        print("[MAIN_V3][ERROR] Quality gate failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print("[MAIN_V3][NEXT] Level 2B quality gate is ready.")

    # ------------------------------------------------------
    # 14) Level 2C — quality compare / visual reports
    # ------------------------------------------------------
    try:
        print("[MAIN_V3] Running Level 2C: quality compare...")

        compare_cfg = QualityCompareConfig(
            max_points=3000,
            dpi=150,
            output_dir_name="Compare",
        )

        quality_compare = QualityCompareRunner(config=compare_cfg)
        compare_artifacts = quality_compare.run(
            resultados_csv=quality_artifacts.resultados_luces_csv,
            schema_by_file_csv=cfg.target_root / "analysis_ready_schema_by_file.csv",
            output_dir=cfg.target_root / "Reports" / "Level1_Quality" / "Compare",
            verbose=True,
        )

        print(f"[MAIN_V3][OK] global compare dir: {compare_artifacts.global_dir}")
        print(f"[MAIN_V3][OK] per-measurement dir: {compare_artifacts.per_measurement_dir}")
        print(f"[MAIN_V3][OK] per-laser dir: {compare_artifacts.per_laser_dir}")
        print(f"[MAIN_V3][OK] per-lab dir: {compare_artifacts.per_lab_dir}")

    except Exception as e:
        print("[MAIN_V3][ERROR] Quality compare failed.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print("[MAIN_V3][NEXT] Level 2C quality compare is ready.")    


    # ------------------------------------------------------
    # 15) Level 3A — correlation lasers config build
    # ------------------------------------------------------
    if CORRELATION_LASERS_ENABLED:
        try:
            print("[MAIN_V3] Running Level 3A: correlation lasers config build...")

            correlation_cfg = CorrelationLasersConfig.from_metrics(
                metrics=CORRELATION_LASERS_METRICS,
                corr_method=CORRELATION_LASERS_CORR_METHOD,
                aggregate_mode=CORRELATION_LASERS_AGGREGATE_MODE,
                require_all_6_lasers=True,
                min_measurements_per_block=3,
                min_valid_lasers_per_measurement=4,
            )

            correlation_cfg_path = (
                cfg.target_root / "correlation_lasers_config.json"
            )
            correlation_cfg.write_json(correlation_cfg_path)

            s = correlation_cfg.summary()

            print(f"[MAIN_V3][OK] correlation_lasers_config.json written to: {correlation_cfg_path}")
            print(f"[MAIN_V3][OK] block_name        : {s['block_name']}")
            print(f"[MAIN_V3][OK] version           : {s['version']}")
            print(f"[MAIN_V3][OK] metrics           : {s['metrics']}")
            print(f"[MAIN_V3][OK] corr_method       : {s['corr_method']}")
            print(f"[MAIN_V3][OK] aggregate_mode    : {s['aggregate_mode']}")
            print(f"[MAIN_V3][OK] expected_lasers   : {s['expected_lasers_count']}")
            print(f"[MAIN_V3][OK] require_all_6     : {s['require_all_6_lasers']}")
            print(f"[MAIN_V3][OK] min_meas_block    : {s['min_measurements_per_block']}")
            print(f"[MAIN_V3][OK] min_valid_lasers  : {s['min_valid_lasers_per_measurement']}")

        except Exception as e:
            print("[MAIN_V3][ERROR] Correlation Lasers config build failed.")
            print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
            return

        print("[MAIN_V3][NEXT] Level 3A correlation lasers config is ready.")

    # ------------------------------------------------------
    # 16) Level 3B — correlation lasers run
    # ------------------------------------------------------
    if CORRELATION_LASERS_RUNNER_ENABLED:
        try:
            print("[MAIN_V3] Running Level 3B: correlation lasers run...")

            correlation_runner = CorrelationLasersRunner(
                root=cfg.target_root,
                config=correlation_cfg,
            )

            correlation_artifacts = correlation_runner.run(
                resultados_csv=quality_artifacts.resultados_luces_csv,
                quality_scores_by_file_csv=quality_artifacts.scores_by_file_csv,
                output_dir=cfg.target_root / "Reports" / "Level3_Correlation_Lasers",
                write=CORRELATION_LASERS_WRITE,
                verbose=CORRELATION_LASERS_VERBOSE,
            )

            print(f"[MAIN_V3][OK] correlation_lasers_base.csv written to: {correlation_artifacts.base_csv}")
            print(f"[MAIN_V3][OK] laser_profiles_by_measurement.csv written to: {correlation_artifacts.profiles_by_measurement_csv}")
            print(f"[MAIN_V3][OK] laser_profile_summary_by_date_lab.csv written to: {correlation_artifacts.summary_by_date_lab_csv}")
            print(f"[MAIN_V3][OK] laser_correlation_by_date_lab.csv written to: {correlation_artifacts.correlation_by_date_lab_csv}")
            print(f"[MAIN_V3][OK] laser_pairwise_dates_by_lab.csv written to: {correlation_artifacts.pairwise_dates_by_lab_csv}")
            print(f"[MAIN_V3][OK] laser_pairwise_labs_by_date.csv written to: {correlation_artifacts.pairwise_labs_by_date_csv}")

            if correlation_artifacts.run_metadata_json is not None:
                print(f"[MAIN_V3][OK] correlation_lasers_run_metadata.json written to: {correlation_artifacts.run_metadata_json}")

        except Exception as e:
            print("[MAIN_V3][ERROR] Correlation Lasers run failed.")
            print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
            return

        print("[MAIN_V3][NEXT] Level 3B correlation lasers run is ready.")        

    # ------------------------------------------------------
    # 17) Level 3C — correlation lasers compare
    # ------------------------------------------------------
    if CORRELATION_LASERS_COMPARE_ENABLED:
        try:
            print("[MAIN_V3] Running Level 3C: correlation lasers compare...")

            correlation_compare = CorrelationLasersCompare(
                config=correlation_cfg,
            )

            correlation_compare_artifacts = correlation_compare.run(
                summary_by_date_lab_csv=correlation_artifacts.summary_by_date_lab_csv,
                correlation_by_date_lab_csv=correlation_artifacts.correlation_by_date_lab_csv,
                pairwise_dates_by_lab_csv=correlation_artifacts.pairwise_dates_by_lab_csv,
                pairwise_labs_by_date_csv=correlation_artifacts.pairwise_labs_by_date_csv,
                profiles_by_measurement_csv=correlation_artifacts.profiles_by_measurement_csv,
                output_dir=cfg.target_root / "Reports" / "Level3_Correlation_Lasers" / "Compare",
                write=CORRELATION_LASERS_COMPARE_WRITE,
                verbose=CORRELATION_LASERS_COMPARE_VERBOSE,
            )

            print(f"[MAIN_V3][OK] correlation global dir: {correlation_compare_artifacts.global_dir}")
            print(f"[MAIN_V3][OK] correlation per-date-lab dir: {correlation_compare_artifacts.per_date_lab_dir}")
            print(f"[MAIN_V3][OK] correlation per-laser dir: {correlation_compare_artifacts.per_laser_dir}")
            print(f"[MAIN_V3][OK] correlation per-lab dir: {correlation_compare_artifacts.per_lab_dir}")
            print(f"[MAIN_V3][OK] global_summary.csv written to: {correlation_compare_artifacts.global_summary_csv}")
            print(f"[MAIN_V3][OK] resumen_global.txt written to: {correlation_compare_artifacts.global_summary_txt}")
            print(f"[MAIN_V3][OK] pairwise_dates_summary.csv written to: {correlation_compare_artifacts.pairwise_dates_summary_csv}")
            print(f"[MAIN_V3][OK] pairwise_labs_summary.csv written to: {correlation_compare_artifacts.pairwise_labs_summary_csv}")

        except Exception as e:
            print("[MAIN_V3][ERROR] Correlation Lasers compare failed.")
            print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
            return

        print("[MAIN_V3][NEXT] Level 3C correlation lasers compare is ready.")            

    # ------------------------------------------------------
    # 18) Level 4A — informational config build
    # ------------------------------------------------------
    if INFORMATIONAL_CONFIG_ENABLED:
        try:
            print("[MAIN_V3] Running Level 4A: informational config build...")

            informational_cfg = InformationalConfig.from_experiment(
                cfg=cfg,
                quality_scores_by_file_path=quality_artifacts.scores_by_file_csv,
                resultados_luces_path=quality_artifacts.resultados_luces_csv,
                level4_folder_name=INFORMATIONAL_LEVEL4_FOLDER_NAME,
                metric_primary=INFORMATIONAL_METRIC_PRIMARY,
                alpha=INFORMATIONAL_ALPHA,
                correction=INFORMATIONAL_CORRECTION,
                coupling_lags_samples=INFORMATIONAL_COUPLING_LAGS,
            )

            preflight = informational_cfg.preflight(
                strict=True,
                parquet_sample_n=25,
                verbose=INFORMATIONAL_CONFIG_VERBOSE,
            )

            informational_paths = informational_cfg.get_level4_paths()
            informational_config_path = informational_paths["level4_config_json"]

            if INFORMATIONAL_CONFIG_WRITE:
                informational_cfg.write_json(informational_config_path)

            print(f"[MAIN_V3][OK] level4_config.json written to: {informational_config_path}")
            print(f"[MAIN_V3][OK] informational mode     : {informational_cfg.mode}")
            print(f"[MAIN_V3][OK] lasers detected        : {len(informational_cfg.lasers)}")
            print(f"[MAIN_V3][OK] lasers source          : {informational_cfg.lasers_source}")
            print(f"[MAIN_V3][OK] bins reference group   : {informational_cfg.bins_reference_group}")
            print(f"[MAIN_V3][OK] preflight ok           : {preflight.ok}")

        except Exception as e:
            print("[MAIN_V3][ERROR] Informational config build failed.")
            print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
            return

        print("[MAIN_V3][NEXT] Level 4A informational config is ready.")           
    # ------------------------------------------------------
    # 19) Level 4B — informational bins build
    # ------------------------------------------------------
    if INFORMATIONAL_BINS_ENABLED:
        try:
            print("[MAIN_V3] Running Level 4B: informational bins build...")

            bins_policy = BinsPolicy()

            bins_artifacts = build_bins_spec_from_config(
                informational_cfg,
                policy=bins_policy,
                reference_dates=None,
                parquet_engine="auto",
            )

            print(f"[MAIN_V3][OK] bins_spec.json written to: {bins_artifacts.bins_spec_json}")
            print(f"[MAIN_V3][OK] bins_summary.csv written to: {bins_artifacts.bins_summary_csv}")

        except Exception as e:
            print("[MAIN_V3][ERROR] Informational bins build failed.")
            print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
            return

        print("[MAIN_V3][NEXT] Level 4B informational bins is ready.")
        
    # ------------------------------------------------------
    # 20) Level 4C — informational bins health
    # ------------------------------------------------------
    if INFORMATIONAL_BINS_HEALTH_ENABLED:
        try:
            print("[MAIN_V3] Running Level 4C: informational bins health...")

            bins_health_policy = BinsHealthPolicy(
                scan_parquets=INFORMATIONAL_BINS_HEALTH_SCAN_PARQUETS,
                parquet_sample_n=INFORMATIONAL_BINS_HEALTH_SAMPLE_N,
                per_parquet_row_cap=INFORMATIONAL_BINS_HEALTH_ROW_CAP,
            )

            bins_health_dir = cfg.target_root / "Reports" / INFORMATIONAL_LEVEL4_FOLDER_NAME / "Bins_Health"

            bins_health_report = evaluate_bins_health(
                bins_spec_json=bins_artifacts.bins_spec_json,
                quality_scores_by_file_csv=quality_artifacts.scores_by_file_csv,
                out_dir=bins_health_dir,
                policy=bins_health_policy,
            )

            if INFORMATIONAL_BINS_HEALTH_VERBOSE:
                print_bins_health_report(bins_health_report)

            print(f"[MAIN_V3][OK] bins_health_report.json written to: {bins_health_dir / 'bins_health_report.json'}")
            print(f"[MAIN_V3][OK] bins_health_per_laser.csv written to: {bins_health_dir / 'bins_health_per_laser.csv'}")
            print(f"[MAIN_V3][OK] bins health status: {bins_health_report.get('status')}")

        except Exception as e:
            print("[MAIN_V3][ERROR] Informational bins health failed.")
            print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
            return

        print("[MAIN_V3][NEXT] Level 4C informational bins health is ready.")
            
            
            
if __name__ == "__main__":
    main()