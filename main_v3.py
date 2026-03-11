from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, List, Tuple
import re
# ==========================================================
# Ensure module path is visible to Python
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
MODULES_DIR = BASE_DIR / "Modulos"

if not MODULES_DIR.exists():
    raise FileNotFoundError(f"Modules directory not found: {MODULES_DIR}")

sys.path.insert(0, str(MODULES_DIR))

# ==========================================================
# Local imports
# ==========================================================
from config import ExperimentConfig


# ==========================================================
# USER PARAMETERS (EDIT HERE)
# ==========================================================
SOURCE_ROOT = BASE_DIR
TARGET_ROOT = BASE_DIR / "Mar26_1"

# Si quieres fijar fechas manualmente, usa por ejemplo:
# SELECTED_DATES = ["02Mar26", "03Mar26", "04Mar26"]
SELECTED_DATES: Optional[List[str]] = None

# Si quieres restringir labs manualmente, usa por ejemplo:
# SELECTED_LABS = ["Betta", "Epsilon"]
SELECTED_LABS: Optional[List[str]] = None

DOUBLE_BLIND = True

# Declarado / externo (por ahora en blanco)
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
# HELPERS
# ==========================================================
def _is_valid_dir(p: Path) -> bool:
    return p.exists() and p.is_dir()

DATE_PATTERN = re.compile(r"^\d{2}[A-Za-z]{3}\d{2}$")

def is_valid_date_folder(name: str) -> bool:
    return bool(DATE_PATTERN.match(name))

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
        raise FileNotFoundError(f"source_root does not exist or is not a directory: {source_root}")

    selected_dates_set = set(selected_dates) if selected_dates is not None else None
    selected_labs_set = set(selected_labs) if selected_labs is not None else None

    dates_found: List[str] = []
    labs_found_set = set()

    for date_dir in sorted(source_root.iterdir()):

        if not date_dir.is_dir():
            continue

        date_name = date_dir.name

        # Ignore folders that are not experiment dates
        if not is_valid_date_folder(date_name):
            continue

        date_name = date_dir.name

        if selected_dates_set is not None and date_name not in selected_dates_set:
            continue

        labs_in_date = [
            p for p in sorted(date_dir.iterdir())
            if p.is_dir() and not p.name.startswith("__")
        ]
        if not labs_in_date:
            continue

        accepted_labs = []
        for lab_dir in labs_in_date:
            lab_name = lab_dir.name
            if selected_labs_set is not None and lab_name not in selected_labs_set:
                continue
            accepted_labs.append(lab_name)

        if accepted_labs:
            dates_found.append(date_name)
            labs_found_set.update(accepted_labs)

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

    if DOUBLE_BLIND:
        label_mode = "blind"
        group_mode = "by_date"
        labels_csv = None
        groups_csv = None
        declared_label_map = {}
        date_to_group = {}
    else:
        # Por ahora dejamos el modo declarado preparado, pero vacío.
        # Ya después lo endurecemos cuando toque ese módulo.
        label_mode = "declared" if DECLARED_LABEL_MAP else "blind"
        group_mode = "declared_ctrl_exp" if DATE_TO_GROUP_DECLARED else "by_date"
        labels_csv = LABELS_CSV_EXTERNAL
        groups_csv = GROUPS_CSV_EXTERNAL
        declared_label_map = DECLARED_LABEL_MAP
        date_to_group = DATE_TO_GROUP_DECLARED

    cfg = ExperimentConfig(
        source_root=SOURCE_ROOT,
        target_root=TARGET_ROOT,
        dates=dates,
        labs=labs,
        default_shift=None,  # <- ya no usamos turno
        rename_map=RENAME_MAP_DECLARED,
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


# ==========================================================
# MAIN
# ==========================================================
def main() -> None:
    try:
        cfg = build_config()
    except Exception as e:
        print("[MAIN_V3][ERROR] Failed to build ExperimentConfig.")
        print(f"[MAIN_V3][ERROR] {type(e).__name__}: {e}")
        return

    print_config_summary(cfg)
    print("[MAIN_V3][OK] Virgin bootstrap completed.")
    print("[MAIN_V3][NEXT] Ready to connect Level 0 in the next iteration.")

if __name__ == "__main__":
    main()