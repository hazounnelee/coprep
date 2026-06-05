r"""Loader for the new consolidated data folder structure.

Expected folder layout::

    데이터폴더/
      통합일지/           ← files with "(\d+)라인" in filename
      수기운전일지/
        1라인/*.xlsx
        2라인/*.xlsx
      용해작업실적/
        MELT_WRK_ORD_METAL.xlsx   (header=1)
        MELT_WRK_ORD.xlsx         (header=1)
      반응투입스케줄/
        반응투입_LOT_INIT.xlsx
        반응투입_LOT_STEP.xlsx
      원재료/
        MATR_{TYPE}_{5chars}_원료검사판정이력_*.xlsx   (3-row header)
"""

import os
import re
import typing as tp

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-material-type property columns to extract.
# Each entry is (category_row_value, item_row_value).
# Column matching is case-insensitive.
MATERIAL_PROPS: dict[str, list[tuple[str, str]]] = {
    "COSO4": [("Chemical Composition (C01)", "Co")],
    "MNSO4": [
        ("Chemical Composition (C01)", "Mn"),
        ("Initial PH (C03)", "pH"),
    ],
    "NISO4": [("Chemical Composition (C01)", "Ni")],
    "NAOH": [("Chemical Composition (C01)", "NaOH")],
    "NH4OH": [],
}

# Regex for extracting material type from filename
_MATERIAL_RE = re.compile(r"MATR_([A-Z0-9]+)_")

# Regex for extracting line number from 통합일지 filename
_LINE_RE = re.compile(r"(\d+)라인")


def _debug(msg: str, debug: bool) -> None:
    if debug:
        print(f"[DEBUG] loader: {msg}")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_integrated(folder: str, debug: bool = False) -> dict[str, pd.DataFrame]:
    """Load 통합일지 files. Returns {line_key: DataFrame}.

    Line key (e.g. "1라인") is extracted from the filename via regex.
    """
    if not os.path.isdir(folder):
        _debug(f"통합일지 folder missing: {folder}", debug)
        return {}

    result: dict[str, pd.DataFrame] = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".xlsx") or fname.startswith("~$"):
            continue
        m = _LINE_RE.search(fname)
        if not m:
            _debug(f"통합일지 skipping file (no line number): {fname}", debug)
            continue
        line_key = f"{m.group(1)}라인"
        path = os.path.join(folder, fname)
        df = pd.read_excel(path, engine="openpyxl")
        result[line_key] = df
        _debug(f"통합일지 {line_key}: {len(df)} rows, {len(df.columns)} cols from {fname}", debug)
    return result


def _load_handrecorded(folder: str, debug: bool = False) -> dict[str, list[pd.DataFrame]]:
    """Load 수기운전일지 files. Returns {line_key: [DataFrame, ...]}.

    Each line subfolder (e.g. "1라인/") contains one or more xlsx files.
    """
    if not os.path.isdir(folder):
        _debug(f"수기운전일지 folder missing: {folder}", debug)
        return {}

    result: dict[str, list[pd.DataFrame]] = {}
    for sub in sorted(os.listdir(folder)):
        sub_path = os.path.join(folder, sub)
        if not os.path.isdir(sub_path):
            continue
        if not sub.endswith("라인"):
            continue
        dfs = []
        for fname in sorted(os.listdir(sub_path)):
            if not fname.endswith(".xlsx") or fname.startswith("~$"):
                continue
            path = os.path.join(sub_path, fname)
            df = pd.read_excel(path, engine="openpyxl")
            dfs.append(df)
        if dfs:
            result[sub] = dfs
            _debug(f"수기운전일지 {sub}: {len(dfs)} files loaded", debug)
    return result


def _load_melt_metal(path: str, debug: bool = False) -> pd.DataFrame:
    """Load MELT_WRK_ORD_METAL.xlsx (header=1, skip row 0)."""
    if not os.path.isfile(path):
        _debug(f"용해_metal file missing: {path}", debug)
        return pd.DataFrame()
    df = pd.read_excel(path, engine="openpyxl", header=1)
    _debug(f"용해_metal: {len(df)} rows, {len(df.columns)} cols", debug)
    return df


def _load_melt_ord(path: str, debug: bool = False) -> pd.DataFrame:
    """Load MELT_WRK_ORD.xlsx (header=1, skip row 0)."""
    if not os.path.isfile(path):
        _debug(f"용해_ord file missing: {path}", debug)
        return pd.DataFrame()
    df = pd.read_excel(path, engine="openpyxl", header=1)
    _debug(f"용해_ord: {len(df)} rows, {len(df.columns)} cols", debug)
    return df


def _load_react_init(path: str, debug: bool = False) -> pd.DataFrame:
    """Load 반응투입_LOT_INIT.xlsx (single header row)."""
    if not os.path.isfile(path):
        _debug(f"반응_init file missing: {path}", debug)
        return pd.DataFrame()
    df = pd.read_excel(path, engine="openpyxl")
    _debug(f"반응_init: {len(df)} rows, {len(df.columns)} cols", debug)
    return df


def _load_react_step(path: str, debug: bool = False) -> pd.DataFrame:
    """Load 반응투입_LOT_STEP.xlsx (single header row)."""
    if not os.path.isfile(path):
        _debug(f"반응_step file missing: {path}", debug)
        return pd.DataFrame()
    df = pd.read_excel(path, engine="openpyxl")
    _debug(f"반응_step: {len(df)} rows, {len(df.columns)} cols", debug)
    return df


def _load_materials(folder: str, debug: bool = False) -> dict[str, pd.DataFrame]:
    """Load 원재료 files with 3-row headers. Returns {TYPE: DataFrame}.

    The 3-row header structure:
        Row 0: main headers (No, 입고일, 입고LOT, category spans...)
        Row 1: categories (검사유형, "Chemical Composition (C01)", ...)
        Row 2: item names (검사항목, Co, Mn, Ni, pH, NaOH, ...)
        Row 3+: actual data

    For each material type, only the columns specified in MATERIAL_PROPS
    plus 입고LOT are extracted. Column matching is case-insensitive.
    """
    if not os.path.isdir(folder):
        _debug(f"원재료 folder missing: {folder}", debug)
        return {}

    # Collect files by material type
    files_by_type: dict[str, list[str]] = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".xlsx") or fname.startswith("~$"):
            continue
        m = _MATERIAL_RE.search(fname)
        if not m:
            continue
        mat_type = m.group(1)
        if mat_type not in MATERIAL_PROPS:
            _debug(f"원재료 unknown type {mat_type}: {fname}", debug)
            continue
        files_by_type.setdefault(mat_type, []).append(
            os.path.join(folder, fname)
        )

    result: dict[str, pd.DataFrame] = {}
    for mat_type, paths in files_by_type.items():
        props = MATERIAL_PROPS[mat_type]
        dfs = []
        for path in paths:
            df = _parse_material_file(path, mat_type, props, debug)
            if df is not None:
                dfs.append(df)
        if dfs:
            result[mat_type] = pd.concat(dfs, ignore_index=True)
            _debug(
                f"원재료 {mat_type}: {len(result[mat_type])} rows total "
                f"from {len(dfs)} files",
                debug,
            )

    return result


def _parse_material_file(
    path: str,
    mat_type: str,
    props: list[tuple[str, str]],
    debug: bool = False,
) -> tp.Optional[pd.DataFrame]:
    """Parse a single material file with a 3-row header.

    Reads all rows starting from row 0 (no header), then uses:
      - row 0 / row 1 as category row
      - row 2 as item-name row
      - row 3+ as data
    """
    raw = pd.read_excel(path, engine="openpyxl", header=None)
    if len(raw) < 3:
        _debug(f"원재료 {mat_type}: file too short: {path}", debug)
        return None

    # Row indices
    category_row = raw.iloc[1].astype(str).str.strip().str.lower()
    item_row = raw.iloc[2].astype(str).str.strip().str.lower()
    data = raw.iloc[3:].reset_index(drop=True)

    # Find 입고LOT column (case-insensitive)
    lot_col_idx = None
    for i, val in enumerate(item_row):
        if val == "입고lot":
            lot_col_idx = i
            break

    if lot_col_idx is None:
        _debug(f"원재료 {mat_type}: 입고LOT column not found in {path}", debug)
        return None

    # Collect wanted column indices
    col_indices = [lot_col_idx]
    col_names = ["입고LOT"]

    for cat, item in props:
        cat_lower = cat.lower()
        item_lower = item.lower()
        found = False
        for i in range(len(item_row)):
            if item_row[i] == item_lower and category_row[i] == cat_lower:
                col_indices.append(i)
                col_names.append(item)
                found = True
                break
        if not found:
            _debug(
                f"원재료 {mat_type}: column ({cat}, {item}) not found in {path}",
                debug,
            )

    result = data.iloc[:, col_indices].copy()
    result.columns = col_names
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_alldata(path_folder: str, debug: bool = False) -> dict:
    """Load all data from the consolidated folder structure.

    Returns:
        {
            "통합일지": {"1라인": df, "2라인": df, ...},
            "수기운전일지": {"1라인": [df, ...], ...},
            "용해_metal": DataFrame,
            "용해_ord": DataFrame,
            "반응_init": DataFrame,
            "반응_step": DataFrame,
            "원재료": {"COSO4": df, "MNSO4": df, ...},
        }
    """
    result: dict = {}

    # 통합일지
    integrated_path = os.path.join(path_folder, "통합일지")
    result["통합일지"] = _load_integrated(integrated_path, debug)

    # 수기운전일지
    handrecorded_path = os.path.join(path_folder, "수기운전일지")
    result["수기운전일지"] = _load_handrecorded(handrecorded_path, debug)

    # 용해작업실적
    melt_dir = os.path.join(path_folder, "용해작업실적")
    result["용해_metal"] = _load_melt_metal(
        os.path.join(melt_dir, "MELT_WRK_ORD_METAL.xlsx"), debug
    )
    result["용해_ord"] = _load_melt_ord(
        os.path.join(melt_dir, "MELT_WRK_ORD.xlsx"), debug
    )

    # 반응투입스케줄
    react_dir = os.path.join(path_folder, "반응투입스케줄")
    result["반응_init"] = _load_react_init(
        os.path.join(react_dir, "반응투입_LOT_INIT.xlsx"), debug
    )
    result["반응_step"] = _load_react_step(
        os.path.join(react_dir, "반응투입_LOT_STEP.xlsx"), debug
    )

    # 원재료
    materials_path = os.path.join(path_folder, "원재료")
    result["원재료"] = _load_materials(materials_path, debug)

    return result
