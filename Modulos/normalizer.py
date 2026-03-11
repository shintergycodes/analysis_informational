"""
normalizer.py

Build a deterministic normalization PLAN from raw discovered files + ExperimentConfig.

This module is intentionally self-contained:
- Depends ONLY on:
    * standard library
    * config.ExperimentConfig
- Does NOT import DatasetIO / io_dataset or any later pipeline modules.
- Does NOT perform IO (no copying, no writing). Planning only.

Supports both modes via ExperimentConfig:
- Declared mode: labels/groups can be attached (declared or external).
- Double-blind mode: labels/groups remain UNK and filenames default to MID-based names
  (unless rename_map explicitly overrides).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ExperimentConfig


# ==========================================================
# Minimal raw-file contract (temporary, local to this module)
# ==========================================================
@dataclass(frozen=True)
class FileItem:
    """
    Minimal descriptor of a discovered raw file.

    This is the smallest contract Normalizer needs to plan normalization.
    A future DatasetIO can produce objects with the same fields.
    """
    date: str
    lab: str
    src_path: Path


# ==========================================================
# Output plan item
# ==========================================================
@dataclass(frozen=True)
class NormalizedItem:
    """
    One planned normalization action.

    - Deterministic mapping: src_path -> dst_path
    - Safe semantics: label/group come from ExperimentConfig modes (declared/blind/external)

    Fields
    ------
    date, lab:
        Identifiers for routing and grouping.
    group:
        "Ctrl" | "Exp" | "UNK" (from cfg.group_for)
    label:
        "LB" | "MEI" | "UNK" (from cfg.label_for)

    mid:
        Neutral measurement id, stable within a run (deterministic based on ordering).
    raw_name:
        Original file name.
    canonical_name:
        Destination file name used under target_root.

    legacy_index:
        Numeric prefix if raw_name matches "<n>med*.csv", else -1
    legacy_color:
        Extracted token only when NOT blind and canonical_name matches "<n>med<token>.csv",
        else "UNK"
    """
    date: str
    lab: str

    src_path: Path
    dst_path: Path

    group: str            # Ctrl / Exp / UNK
    label: str            # LB / MEI / UNK

    mid: str
    raw_name: str
    canonical_name: str

    legacy_index: int     # parseable <n>med..., else -1
    legacy_color: str     # meaningful only when declared naming encodes it, else UNK


# ==========================================================
# Normalizer
# ==========================================================
class Normalizer:
    """
    Semantic translator:
        List[FileItem] + ExperimentConfig  ->  List[NormalizedItem]

    No inference:
      - Does not infer label/group from filenames or data.
      - Uses cfg.label_for / cfg.group_for only.

    Deterministic:
      - Sorts items before assigning fallback indices for MID generation.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        dst_subdir: str = "Raw Data",
        prefer_mid_filenames_in_blind: bool = True,
        mid_format: str = "{date}_{lab}_{index:03d}",
    ) -> None:
        self.cfg = cfg
        self.dst_subdir = dst_subdir
        self.prefer_mid_filenames_in_blind = prefer_mid_filenames_in_blind
        self.mid_format = mid_format

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def plan(self, items: List[FileItem]) -> List[NormalizedItem]:
        """
        Build the full normalization plan (no IO).

        Parameters
        ----------
        items:
            Raw discovered files.

        Returns
        -------
        List[NormalizedItem]
            Deterministic plan describing src->dst actions and safe metadata.
        """
        if not items:
            return []

        ordered = sorted(items, key=lambda it: (it.date, it.lab, it.src_path.name))

        # Fallback counters per (date, lab) for stable MID indexing when raw names
        # do not contain a numeric prefix.
        counters: Dict[Tuple[str, str], int] = {}

        out: List[NormalizedItem] = []

        for item in ordered:
            raw_name = item.src_path.name

            # Determine a stable index for MID generation
            parsed_idx = self._extract_index_optional(raw_name)
            if parsed_idx is None:
                key = (item.date, item.lab)
                counters[key] = counters.get(key, 0) + 1
                idx_for_mid = counters[key]
            else:
                idx_for_mid = parsed_idx

            mid = self._build_mid(date=item.date, lab=item.lab, index=idx_for_mid)

            canonical_name = self._choose_canonical_name(raw_name=raw_name, mid=mid)
            dst_path = self._dst_path(date=item.date, lab=item.lab, dst_name=canonical_name)

            group = self.cfg.group_for(date=item.date, mid=mid, canonical_name=canonical_name)
            label = self.cfg.label_for(mid=mid, canonical_name=canonical_name)

            legacy_index = parsed_idx if parsed_idx is not None else -1

            legacy_color = "UNK"
            if self.cfg.label_mode != "blind":
                legacy_color = self._extract_color_optional(canonical_name) or "UNK"

            out.append(
                NormalizedItem(
                    date=item.date,
                    lab=item.lab,
                    src_path=item.src_path,
                    dst_path=dst_path,
                    group=group,
                    label=label,
                    mid=mid,
                    raw_name=raw_name,
                    canonical_name=canonical_name,
                    legacy_index=legacy_index,
                    legacy_color=legacy_color,
                )
            )

        return out

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _dst_path(self, *, date: str, lab: str, dst_name: str) -> Path:
        return self.cfg.target_root / date / lab / self.dst_subdir / dst_name

    def _choose_canonical_name(self, *, raw_name: str, mid: str) -> str:
        # 1) If rename_map provides a canonical name, always respect it (both modes)
        canon = self.cfg.canonical_name_for(raw_name)
        if canon != raw_name:
            return canon

        # 2) If blind, prefer MID-based filenames to avoid embedding semantics in names
        if self.cfg.label_mode == "blind" and self.prefer_mid_filenames_in_blind:
            return f"{mid}.csv"

        # 3) Otherwise keep raw filename
        return raw_name

    def _build_mid(self, *, date: str, lab: str, index: int) -> str:
        try:
            return self.mid_format.format(
                date=date,
                lab=lab,
                index=int(index),
            )
        except Exception:
            return f"{date}_{lab}_{int(index):03d}"

    @staticmethod
    def _extract_index_optional(name: str) -> Optional[int]:
        """
        Extract numeric prefix before 'med' for names like '<n>med*.csv'.
        Returns None if pattern does not match.
        """
        try:
            if "med" not in name:
                return None
            prefix = name.split("med", 1)[0].strip()
            if not prefix:
                return None
            return int(prefix)
        except Exception:
            return None

    @staticmethod
    def _extract_color_optional(name: str) -> Optional[str]:
        """
        Extract token between 'med' and '.csv' ONLY if name matches '<n>med<token>.csv'.
        Returns None if pattern does not match.
        """
        try:
            if not name.endswith(".csv"):
                return None
            if "med" not in name:
                return None
            stem = name[:-4]
            left, right = stem.split("med", 1)
            if not left.isdigit():
                return None
            token = right.strip()
            return token or None
        except Exception:
            return None
