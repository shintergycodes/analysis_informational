# manifest.py
r"""
Module 4 — Manifest (minimal registry for analysis)

Responsibility:
- Record, in a structured and minimal way, the result of the preparation pipeline.
- The manifest is the formal evidence of which files were prepared and
  with what experimental semantics (if available).

This module:
- DOES NOT discover files.
- DOES NOT decide renames.
- DOES NOT copy files.
- ONLY records what was already decided and (optionally) executed.

Source of truth:
- Iterable[NormalizedItem] produced by the Normalizer.

Output:
- manifest_all.csv with minimal columns for downstream analysis.

Compatibility notes:
- This module is compatible with the *new* NormalizedItem contract produced by the
  updated normalizer.py:
    NormalizedItem.date        -> manifest 'fecha'
    NormalizedItem.group       -> manifest 'jornada'
    NormalizedItem.label       -> manifest 'etiqueta'
    NormalizedItem.legacy_color-> manifest 'color'
    NormalizedItem.legacy_index-> manifest 'archivo'

- In double-blind mode, group/label/color may be "UNK" and legacy_index may be -1.
  The manifest preserves these values (no inference/filling).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from typing import Optional

from config import ExperimentConfig
from normalizer import NormalizedItem


@dataclass(frozen=True)
class ManifestRow:
    """
    Minimal row in the manifest.

    Methodological note:
    - 'archivo' is a technical id used in some legacy analyses; in double-blind
      runs it can be -1 (unknown / not meaningful).
    - Full filesystem paths are not stored to avoid coupling analysis to the filesystem.
    """
    fecha: str
    jornada: str
    lab: str
    turno: Optional[str]
    color: str
    etiqueta: str
    archivo: int


class ManifestWriter:
    """
    Manifest writer: NormalizedItem -> ManifestRow -> CSV.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    # ---------------------------------------------------------
    # Main API
    # ---------------------------------------------------------
    def build_rows(self, plan: Iterable[NormalizedItem]) -> List[ManifestRow]:
        """
        Build manifest rows from the normalization plan.
        """
        rows: List[ManifestRow] = []

        for ni in plan:
            rows.append(
                ManifestRow(
                    fecha=ni.date,
                    jornada=ni.group,
                    lab=ni.lab,
                    turno=None,
                    color=ni.legacy_color,
                    etiqueta=ni.label,
                    archivo=ni.legacy_index,
                )
            )

        return rows

    def write(self, plan: Iterable[NormalizedItem], output_path: Path) -> None:
        """
        Write manifest_all.csv.

        Precondition:
        - The plan has been executed (copies performed), or at least validated
          as correct.
        """
        rows = self.build_rows(plan)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["fecha", "jornada", "lab", "turno", "color", "etiqueta", "archivo"]
            )
            for r in rows:
                writer.writerow(
                    [
                        r.fecha,
                        r.jornada,
                        r.lab,
                        r.turno,
                        r.color,
                        r.etiqueta,
                        r.archivo,
                    ]
                )
