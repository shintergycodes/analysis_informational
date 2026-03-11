# summary_table.py
"""
summary_table.py

Human-readable summary of manifest_all.csv.

Purpose
-------
Provide a compact structural overview of the prepared dataset,
independent of experimental semantics.

This module is designed to be useful in:
- double-blind mode
- declared mode
- partially unblinded runs

It performs NO inference and NO data transformation.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List


def load_manifest(manifest_path: Path) -> List[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def summarize(manifest_rows: List[dict]) -> Dict[Tuple[str, str], int]:
    """
    Aggregate by (fecha, lab).
    """
    counter: Dict[Tuple[str, str], int] = defaultdict(int)

    for row in manifest_rows:
        key = (
            row["fecha"],
            row["lab"],
        )
        counter[key] += 1

    return counter

def print_summary(summary: Dict[Tuple[str, str], int]) -> None:
    headers = ["Fecha", "Laboratorio", "Archivos"]

    rows = [
        (fecha, lab, str(count))
        for (fecha, lab), count in sorted(summary.items())
    ]

    col_widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]

    def sep() -> str:
        return "+".join("-" * (w + 2) for w in col_widths)

    def fmt(row: Tuple[str, ...]) -> str:
        return "|".join(f" {row[i]:<{col_widths[i]}} " for i in range(len(row)))

    print(sep())
    print(fmt(tuple(headers)))
    print(sep())
    for r in rows:
        print(fmt(r))
    print(sep())

def main() -> None:
    manifest_path = Path("manifest_all.csv")

    if not manifest_path.exists():
        print("[SUMMARY][ERROR] manifest_all.csv not found.")
        return

    rows = load_manifest(manifest_path)
    summary = summarize(rows)

    print("[SUMMARY] Dataset structure summary:")
    print_summary(summary)


if __name__ == "__main__":
    main()
