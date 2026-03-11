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
# EXP SEPTUP (from Modulos Python)
# ==========================================================
from config import ExperimentConfig
from io_dataset import DatasetIO
from normalizer import Normalizer
from manifest import ManifestWriter
from summary_table import load_manifest, summarize, print_summary

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
    
if __name__ == "__main__":
    main()