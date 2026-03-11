# io_dataset.py
"""
io_dataset.py

Dataset discovery + minimal filesystem utilities.

Design goals
------------
- Only discovers what exists on disk (no experimental inference).
- Produces a minimal FileItem contract that downstream modules can consume.
- Supports both declared and double-blind runs (independent of semantics).
- Can execute a copy plan produced elsewhere (e.g., Normalizer).

Expected directory layout (flexible)
------------------------------------
Primary (recommended):
  <source_root>/<date>/<lab>/Raw Data/*.csv

Fallback (optional):
  <source_root>/<date>/<lab>/*.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set
import shutil

from config import ExperimentConfig


@dataclass(frozen=True)
class FileItem:
    """
    Minimal descriptor of a discovered raw CSV file.
    This object is intentionally semantic-free (no LB/MEI, no Ctrl/Exp).
    """
    date: str
    lab: str
    src_path: Path
    container_dir: Path  # directory where it was found (lab root or Raw Data)

    @property
    def src_name(self) -> str:
        return self.src_path.name


class DatasetIO:
    """
    Disk discovery for the raw dataset.

    Responsibilities
    ---------------
    - Enumerate labs under <source_root>/<date>/
    - Enumerate CSV files under:
        - <lab>/Raw Data/ (default)
        - optionally also <lab>/ (lab root)
    - Build FileItem list for downstream planning (Normalizer).

    Non-responsibilities
    --------------------
    - No renaming decisions
    - No label/group assignment
    - No data loading/parsing
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        raw_subdir: str = "Raw Data",
        include_lab_root_csv: bool = True,
    ) -> None:
        self.cfg = cfg
        self.raw_subdir = raw_subdir
        self.include_lab_root_csv = include_lab_root_csv

    # -------------------------
    # Path helpers
    # -------------------------
    def date_dir(self, date: str) -> Path:
        return self.cfg.source_root / date

    def lab_dir(self, date: str, lab: str) -> Path:
        return self.date_dir(date) / lab

    def raw_dir(self, date: str, lab: str) -> Path:
        return self.lab_dir(date, lab) / self.raw_subdir

    # -------------------------
    # Discovery
    # -------------------------
    def list_labs(self, date: str, *, strict: bool = False) -> List[str]:
        base = self.date_dir(date)
        if not base.exists():
            if strict:
                raise FileNotFoundError(f"Date directory does not exist: {base}")
            return []

        labs: List[str] = []
        for p in base.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith("__") or p.name.startswith("."):
                continue
            labs.append(p.name)

        return sorted(labs)

    def list_csv_paths(self, date: str, lab: str, *, strict: bool = False) -> List[Path]:
        """
        Returns a deduplicated, sorted list of CSV paths for a (date, lab).

        Search locations:
          - <lab>/Raw Data/*.csv (primary)
          - optionally <lab>/*.csv (fallback)
        """
        lab_dir = self.lab_dir(date, lab)
        if not lab_dir.exists():
            if strict:
                raise FileNotFoundError(f"Lab directory does not exist: {lab_dir}")
            return []

        candidates: List[Path] = []

        rd = self.raw_dir(date, lab)
        if rd.exists() and rd.is_dir():
            candidates.extend([p for p in rd.glob("*.csv") if p.is_file()])
        elif strict:
            raise FileNotFoundError(f"Raw Data directory does not exist: {rd}")

        if self.include_lab_root_csv:
            candidates.extend([p for p in lab_dir.glob("*.csv") if p.is_file()])

        # Deduplicate by resolved absolute path (avoid duplicates if symlinks/overlaps)
        seen: Set[Path] = set()
        unique: List[Path] = []
        for p in candidates:
            try:
                key = p.resolve()
            except Exception:
                key = p.absolute()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return sorted(unique, key=lambda x: x.name)

    def collect_date(self, date: str, *, strict: bool = False) -> List[FileItem]:
        items: List[FileItem] = []
        for lab in self.list_labs(date, strict=strict):
            lab_dir = self.lab_dir(date, lab)
            rd = self.raw_dir(date, lab)

            csvs = self.list_csv_paths(date, lab, strict=strict)
            for p in csvs:
                container = rd if (rd.exists() and rd.is_dir() and rd in p.parents) else lab_dir
                items.append(FileItem(date=date, lab=lab, src_path=p, container_dir=container))
        return items

    def collect_all(self, *, strict: bool = False) -> List[FileItem]:
        """
        Collect FileItem for all dates declared in ExperimentConfig.

        Note:
          Uses cfg.dates (v2). This is the authoritative list of dates to process.
        """
        items: List[FileItem] = []
        for date in list(self.cfg.dates):
            items.extend(self.collect_date(date, strict=strict))
        return items

    # -------------------------
    # Quick reporting
    # -------------------------
    def quick_report(self, items: Iterable[FileItem], *, max_show: int = 5) -> str:
        """
        Human-readable summary by (date, lab).
        """
        from collections import defaultdict

        grouped = defaultdict(list)
        for it in items:
            grouped[(it.date, it.lab)].append(it)

        lines: List[str] = []
        for (date, lab) in sorted(grouped.keys()):
            group = grouped[(date, lab)]
            lines.append(f"{date} | {lab} -> {len(group)} CSVs")
            for it in group[:max_show]:
                lines.append(f"  - {it.src_name}")
            if len(group) > max_show:
                lines.append("  ...")
        return "\n".join(lines)

    # -------------------------
    # Physical execution (copy)
    # -------------------------
    def execute_plan(self, plan: Iterable, *, dry_run: bool = False, overwrite: bool = True) -> None:
        """
        Execute a copy plan that provides .src_path and .dst_path for each item.

        - Creates destination directories if missing.
        - Uses shutil.copy2 to preserve metadata.
        - Overwrites by default (overwrite=True).
        """
        for ni in plan:
            src_path = getattr(ni, "src_path", None)
            dst_path = getattr(ni, "dst_path", None)
            if src_path is None or dst_path is None:
                raise TypeError("Plan items must expose 'src_path' and 'dst_path' attributes")

            dst_dir = Path(dst_path).parent
            dst_dir.mkdir(parents=True, exist_ok=True)

            if dry_run:
                print(f"[DRY-RUN] {src_path} -> {dst_path}")
                continue

            if (not overwrite) and Path(dst_path).exists():
                continue

            shutil.copy2(Path(src_path), Path(dst_path))
