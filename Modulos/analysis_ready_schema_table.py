# analysis_ready_schema_table.py
"""
Analysis Ready Schema Table

Purpose
-------
Validate schema consistency of Analysis Ready Parquet files against
the canonical column roles definition.

This module:
- Works in both double-blind and declared experimental modes
- Does NOT assume knowledge of intervention day or label semantics
- Uses `mid` as the unique identifier for Parquet files
- Does NOT modify data; schema inspection only

Outputs
-------
1) analysis_ready_schema_by_file.csv
2) analysis_ready_schema_table.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import json
import hashlib

import pandas as pd


# ==========================================================
# Helpers
# ==========================================================
def _hash_schema(columns: List[str]) -> str:
    joined = "|".join(columns)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _load_expected_schema(role_entry: Dict[str, Any]) -> List[str]:
    roles = role_entry.get("roles", {})
    columns: List[str] = []

    if "t_sys" in roles and roles["t_sys"]:
        columns.append("t_sys")
    if "t_rel" in roles and roles["t_rel"]:
        columns.append("t_rel")

    for ch in roles.get("channels", []):
        cname = ch.get("canonical_name")
        if cname:
            columns.append(str(cname))

    return columns


# ==========================================================
# Main API
# ==========================================================
def build_analysis_ready_schema_tables(
    *,
    catalog_df: pd.DataFrame,
    column_roles_path: Path,
    analysis_ready_root: Path,
    output_root: Path,
) -> None:
    """
    Build schema validation tables for Analysis Ready Parquet files.

    Parameters
    ----------
    catalog_df:
        Analysis-ready catalog (one row per measurement).
        Required columns:
        - mid
        - fecha
        - lab

        Optional legacy/informative columns:
        - jornada
        - turno
        - etiqueta

    column_roles_path:
        Path to column_roles.json.

    analysis_ready_root:
        Root directory where Analysis Ready Parquets are stored.

    output_root:
        Directory where CSV reports will be written.
    """

    # ------------------------------------------------------
    # Load roles
    # ------------------------------------------------------
    with Path(column_roles_path).open("r", encoding="utf-8") as f:
        roles_map = json.load(f)

    rows_by_file: List[Dict[str, Any]] = []

    # ------------------------------------------------------
    # Per-file schema validation
    # ------------------------------------------------------
    for row in catalog_df.itertuples(index=False):
        mid = str(getattr(row, "mid"))
        fecha = str(getattr(row, "fecha"))
        lab = str(getattr(row, "lab"))
        jornada = str(getattr(row, "jornada", "UNK"))
        turno = getattr(row, "turno", "UNK")
        etiqueta = getattr(row, "etiqueta", "UNK")

        parquet_path = (
            Path(analysis_ready_root)
            / fecha
            / lab
            / f"{mid}.parquet"
        )

        role_entry = roles_map.get(mid)
        if role_entry is None or role_entry.get("status") != "ok":
            expected_cols = []
        else:
            expected_cols = _load_expected_schema(role_entry)

        expected_hash = _hash_schema(expected_cols)

        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, engine="pyarrow")
                actual_cols = list(df.columns)
                parquet_exists = True
            except Exception:
                actual_cols = []
                parquet_exists = False
        else:
            actual_cols = []
            parquet_exists = False

        actual_hash = _hash_schema(actual_cols)
        schema_ok = parquet_exists and (expected_cols == actual_cols)

        rows_by_file.append(
            {
                "mid": mid,
                "fecha": fecha,
                "lab": lab,
                "turno": turno,
                "jornada": jornada,
                "etiqueta": etiqueta,
                "parquet_exists": parquet_exists,
                "schema_ok": schema_ok,
                "expected_schema": ",".join(expected_cols),
                "actual_schema": ",".join(actual_cols),
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
            }
        )

    df_by_file = pd.DataFrame(rows_by_file)

    # ------------------------------------------------------
    # Group-level aggregation
    # ------------------------------------------------------
    group_cols = ["fecha", "lab"]
    agg_rows: List[Dict[str, Any]] = []


    for keys, g in df_by_file.groupby(group_cols):
        fecha, lab = keys

        total = len(g)
        ok_count = int(g["schema_ok"].sum())
        unique_hashes = sorted(set(g["actual_hash"]))

        etiquetas_presentes = sorted(set(g["etiqueta"]) - {"UNK"})
        mixed_labels = len(etiquetas_presentes) > 1

        agg_rows.append(
            {
                "fecha": fecha,
                "lab": lab,
                "n_files": total,
                "n_schema_ok": ok_count,
                "n_schema_fail": total - ok_count,
                "n_schema_variants": len(unique_hashes),
                "schema_consistent": len(unique_hashes) == 1,
                "mixed_labels": mixed_labels,
                "labels_present": ",".join(etiquetas_presentes),
            }
        )

    df_group = pd.DataFrame(agg_rows)

    # ------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df_by_file.to_csv(
        output_root / "analysis_ready_schema_by_file.csv",
        index=False,
    )

    df_group.to_csv(
        output_root / "analysis_ready_schema_table.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Console warnings (non-fatal)
    # ------------------------------------------------------
    if not df_group.empty:
        problematic = df_group[
            (~df_group["schema_consistent"]) | (df_group["n_schema_fail"] > 0)
        ]

        if not problematic.empty:
            print("[SCHEMA][WARN] Schema inconsistencies detected:")
            print(
                #
                problematic[
                    [
                        "fecha",
                        "lab",
                        "n_schema_fail",
                        "n_schema_variants",
                        "mixed_labels",
                    ]
                ]
            )
        else:
            print("[SCHEMA][OK] All Analysis Ready schemas are consistent.")
