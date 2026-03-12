# analysis_table_io.py
"""
Analysis Table IO

Purpose
-------
Provide low-level, low-memory IO utilities for AnalysisTableBuilder:
- Read a CSV in streaming mode
- Optionally drop a trailing "extra field" to align row width with header indices
- Select columns by index (not by name)
- Convert channel columns to float and apply sentinel->NaN
- Produce either:
    (A) a pandas DataFrame (fallback / small files)
    (B) a Parquet file directly (preferred for large files when pyarrow is available)

This module is intentionally independent from experimental semantics and from
the pipeline orchestrator. It is a pure IO/transform layer.

Contracts
---------
Inputs are strictly positional:
- indices: list[int] selecting columns from the raw CSV row
- out_columns: list[str] canonical output column names aligned with indices
- channel_columns: list[str] subset of out_columns that should be treated as numeric channels
- sentinels: set[float] values to replace with NaN in channel columns

Memory / Performance
--------------------
- Streaming CSV read via csv.reader
- Optional streaming Parquet write via pyarrow.parquet.ParquetWriter
- Fallback to pandas in-memory DataFrame + df.to_parquet for environments without pyarrow

No global state; single-responsibility functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import csv
import math

import pandas as pd


_DEFAULT_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")


# -----------------------------
# Public API
# -----------------------------
def build_dataframe_from_csv(
    *,
    csv_path: Path,
    indices: Sequence[int],
    out_columns: Sequence[str],
    channel_columns: Sequence[str],
    sentinels: Set[float],
    delimiter: str = ",",
    encoding: Optional[str] = None,
    encoding_candidates: Sequence[str] = _DEFAULT_ENCODINGS,
    drop_extra_field: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Read a CSV and return a DataFrame with selected/converted columns.

    This is a fallback path for small/moderate files or when streaming Parquet write
    is not desired. It still reads CSV in streaming mode but materializes all rows
    in memory at the end.

    Returns:
        (df, stats) where stats contains:
          - rows_written
          - bad_rows
    """
    _validate_args(indices, out_columns, channel_columns)

    rows: List[List[float]] = []
    bad_rows = 0
    max_idx = max(indices) if indices else -1

    f = _open_text(csv_path, encoding, encoding_candidates)
    with f:
        reader = csv.reader(f, delimiter=delimiter)
        header_n = _skip_header(reader)
        if header_n is None:
            df_empty = pd.DataFrame(columns=list(out_columns))
            return df_empty, {"rows_written": 0, "bad_rows": 0}

        for raw_row in reader:
            row = _normalize_row(raw_row, header_n, drop_extra_field)
            if row is None:
                continue
            if max_idx >= len(row):
                bad_rows += 1
                continue

            extracted = [row[i] for i in indices]
            converted = convert_extracted_row(
                extracted=extracted,
                out_columns=out_columns,
                channel_columns=set(channel_columns),
                sentinels=sentinels,
            )
            rows.append(converted)

    df = pd.DataFrame(rows, columns=list(out_columns))
    return df, {"rows_written": len(df), "bad_rows": bad_rows}


def write_parquet_from_csv(
    *,
    csv_path: Path,
    parquet_path: Path,
    indices: Sequence[int],
    out_columns: Sequence[str],
    channel_columns: Sequence[str],
    sentinels: Set[float],
    delimiter: str = ",",
    encoding: Optional[str] = None,
    encoding_candidates: Sequence[str] = _DEFAULT_ENCODINGS,
    drop_extra_field: bool = False,
    row_group_size: int = 50_000,
) -> Dict[str, int]:
    """
    Preferred path for large files: stream CSV -> stream Parquet using pyarrow.
    Raises ImportError if pyarrow is not available.

    Returns stats:
      - rows_written
      - bad_rows
    """
    _validate_args(indices, out_columns, channel_columns)

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise ImportError("pyarrow is required for streaming Parquet writing.") from e

    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        parquet_path.unlink(missing_ok=True)

    max_idx = max(indices) if indices else -1
    bad_rows = 0
    rows_written = 0

    # Output schema: float64 for all columns (times + channels)
    schema = pa.schema([(str(name), pa.float64()) for name in out_columns])

    channel_set = set(channel_columns)

    # Pre-allocate buffers
    buffers: Dict[str, List[float]] = {str(c): [] for c in out_columns}

    def flush(writer: Any) -> int:
        nonlocal buffers
        n = len(buffers[str(out_columns[0])]) if out_columns else 0
        if n <= 0:
            return 0
        arrays = [pa.array(buffers[str(c)], type=pa.float64()) for c in out_columns]
        table = pa.Table.from_arrays(arrays, schema=schema)
        #table = pa.Table.from_arrays(arrays, names=[str(c) for c in out_columns], schema=schema)
        writer.write_table(table)
        buffers = {str(c): [] for c in out_columns}
        return n

    f = _open_text(csv_path, encoding, encoding_candidates)
    with f:
        reader = csv.reader(f, delimiter=delimiter)
        header_n = _skip_header(reader)
        if header_n is None:
            return {"rows_written": 0, "bad_rows": 0}

        with pq.ParquetWriter(parquet_path, schema=schema) as writer:
            for raw_row in reader:
                row = _normalize_row(raw_row, header_n, drop_extra_field)
                if row is None:
                    continue
                if max_idx >= len(row):
                    bad_rows += 1
                    continue

                extracted = [row[i] for i in indices]
                converted = convert_extracted_row(
                    extracted=extracted,
                    out_columns=out_columns,
                    channel_columns=channel_set,
                    sentinels=sentinels,
                )

                for c, v in zip(out_columns, converted):
                    buffers[str(c)].append(v)

                if len(buffers[str(out_columns[0])]) >= max(1, int(row_group_size)):
                    rows_written += flush(writer)

            rows_written += flush(writer)

    return {"rows_written": rows_written, "bad_rows": bad_rows}


def convert_extracted_row(
    *,
    extracted: Sequence[Any],
    out_columns: Sequence[str],
    channel_columns: Set[str],
    sentinels: Set[float],
) -> List[float]:
    """
    Convert an extracted row (already selected by index) into floats.

    Rules:
    - All outputs are float64.
    - For channel columns only:
        - sentinel values -> NaN
    - For time columns (or any non-channel):
        - attempt float conversion; failure -> NaN
      (Time string parsing is intentionally NOT done here; time bounds are preserved
       separately by measurement_time_bounds.json)
    """
    out: List[float] = []
    for name, raw in zip(out_columns, extracted):
        v = _to_float(raw)
        if str(name) in channel_columns:
            if (not math.isnan(v)) and (v in sentinels):
                v = float("nan")
        out.append(v)
    return out


# -----------------------------
# Internals
# -----------------------------
def _validate_args(
    indices: Sequence[int],
    out_columns: Sequence[str],
    channel_columns: Sequence[str],
) -> None:
    if not indices:
        raise ValueError("indices must be non-empty.")
    if len(indices) != len(out_columns):
        raise ValueError("indices and out_columns must have the same length.")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must be unique.")
    if len(set(out_columns)) != len(out_columns):
        raise ValueError("out_columns must be unique.")
    # channel_columns must be a subset of out_columns (if not, it is a contract bug upstream)
    oc = set(map(str, out_columns))
    cc = set(map(str, channel_columns))
    if not cc.issubset(oc):
        missing = sorted(list(cc - oc))
        raise ValueError(f"channel_columns must be subset of out_columns. Missing: {missing}")


def _open_text(path: Path, encoding: Optional[str], encodings: Sequence[str]):
    """
    Open a text file with preferred encoding and fallbacks.
    """
    p = Path(path)
    candidates: List[str] = []
    if encoding and str(encoding).strip():
        candidates.append(str(encoding).strip())
    for e in encodings:
        e2 = str(e).strip()
        if e2 and e2 not in candidates:
            candidates.append(e2)

    last_err: Optional[Exception] = None
    for enc in candidates:
        try:
            return p.open("r", encoding=enc, newline="")
        except Exception as e:
            last_err = e
    raise last_err or OSError(f"Unable to open CSV: {p}")


def _skip_header(reader: csv.reader) -> Optional[int]:
    """
    Read and discard the header row, returning header column count.
    Returns None for empty files.
    """
    try:
        header = next(reader)
    except StopIteration:
        return None
    return len(header) if header is not None else None


def _normalize_row(raw_row: List[str], header_n: int, drop_extra_field: bool) -> Optional[List[str]]:
    """
    Normalize a raw csv.reader row:
    - skip empty rows
    - optionally drop trailing extra field
      *safe rule*: drop only if row has header_n + 1 fields
    - return None if row does not match header width after normalization
    """
    if not raw_row:
        return None

    row = raw_row
    if drop_extra_field and len(row) == header_n + 1:
        row = row[:-1]

    if len(row) != header_n:
        return None
    return row


def _to_float(x: Any) -> float:
    """
    Robust float conversion. Non-numeric -> NaN.
    """
    if x is None:
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    low = s.lower()
    if low in ("nan", "none", "null"):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")
