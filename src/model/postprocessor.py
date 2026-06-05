import re
import typing as tp
import pandas as pd
import numpy as np

LIST_PREFIX = ["STEP_WEIGHT_", "STEP_RPM_", "STEP_PH_", "STEP_SIZE_", "STEP_METERIAL_"]
PATTERN_METERIAL = re.compile(r"^STEP_METERIAL_(\d+)")
PATTERN_GENERAL = re.compile(r"_(\d+)$")

MATERIAL_FILTER_TERMS = [
    "_황산망간_성분_ChemicalComposition(C01)_Mn",
    "_황산망간_성분_InitialPH(C03)_pH",
    "_황산니켈_성분_ChemicalComposition(C01)_ni",
    "_가성소다_성분_ChemicalComposition(C01)_NaOH",
    "_황산코발트_투입량",
]


def split_xy(df: pd.DataFrame) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    size_cols = [c for c in df.columns if c.startswith("STEP_SIZE_")]
    x = df[[c for c in df.columns if c not in size_cols]]
    y = df[size_cols]
    return x, y


def _build_step_col_map(df: pd.DataFrame, include_material: bool) -> dict:
    prefixes = LIST_PREFIX if include_material else LIST_PREFIX[:-1]
    col_map = {p: [] for p in prefixes}
    for col in df.columns:
        n = None
        if col.startswith("STEP_METERIAL_"):
            m = PATTERN_METERIAL.match(col)
            if m:
                n = int(m.group(1))
        if n is None:
            m = PATTERN_GENERAL.search(col)
            if m:
                n = int(m.group(1))
        if n is not None and 1 <= n <= 60:
            for prefix in prefixes:
                if col.startswith(prefix):
                    col_map[prefix].append((n, col))
                    break
    for p in col_map:
        col_map[p].sort(key=lambda x: x[0])
    return col_map


def _get_max_step(row: pd.Series, df: pd.DataFrame) -> int:
    """Determine max valid step for a row.

    Prefers STEP_SIZE_DMAX columns; falls back to any STEP_SIZE_ columns
    when DMAX columns are absent (e.g. in test fixtures using D50 only).
    """
    check_cols = [c for c in df.columns if c.startswith("STEP_SIZE_DMAX")]
    if not check_cols:
        # Fallback: use any STEP_SIZE_ columns to infer step count
        check_cols = [c for c in df.columns if c.startswith("STEP_SIZE_")]
    return int(row[check_cols].dropna().__len__())


def augment_data(df: pd.DataFrame, include_material: bool = False) -> pd.DataFrame:
    """배치당 max_step 행 생성. 미래 스텝 값은 0.0으로 마스킹."""
    init_cols = ["lot_target", "lot_reacted"] + [
        c for c in df.columns if c.startswith("반응투입_초기조건_")
    ]
    col_map = _build_step_col_map(df, include_material)

    rows = []
    for _, row in df.iterrows():
        max_step = _get_max_step(row, df)
        for j in range(max_step):
            new_row = {c: row[c] for c in init_cols if c in row.index}
            for prefix, step_cols in col_map.items():
                for n, col in step_cols:
                    new_row[col] = row[col] if n <= j + 1 else 0.0
            rows.append(new_row)

    all_cols = init_cols + [col for p in col_map for _, col in col_map[p]]
    return pd.DataFrame(rows, columns=[c for c in all_cols if c in df.columns])


def filter_material(df_x: pd.DataFrame) -> pd.DataFrame:
    """원료 물성 컬럼을 핵심 항목만 남김."""
    mat_cols = df_x.filter(like="STEP_METERIAL").columns.tolist()
    keep = [c for c in mat_cols if any(t in c for t in MATERIAL_FILTER_TERMS)]
    drop = list(set(mat_cols) - set(keep))
    return df_x.drop(columns=drop)


def postprocess_by_product(
    df_x: pd.DataFrame, df_y: pd.DataFrame, product_type: str
) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """대/소입경 후처리."""
    if product_type == "소입경":
        bad = df_y[(df_y["STEP_SIZE_DMIN_02"].isna()) | (df_y["STEP_SIZE_DMIN_02"] == 0)].index
        df_y = df_y.drop(columns=df_y.columns[:5])
        df_x, df_y = df_x.drop(bad), df_y.drop(bad)
        df_x, df_y = df_x.reset_index(drop=True), df_y.reset_index(drop=True)

        for prefix in ["STEP_RPM_", "STEP_PH_", "STEP_METERIAL_"]:
            drop_cols = df_x.filter(like=f"{prefix}01").columns
            df_x = df_x.drop(columns=drop_cols)
        for base in ["STEP_WEIGHT_Metal", "STEP_WEIGHT_NaOH", "STEP_WEIGHT_NH4OH"]:
            col1, col2 = f"{base}_01", f"{base}_02"
            if col1 in df_x and col2 in df_x:
                df_x[col2] = df_x[col2] + df_x[col1]
                df_x = df_x.drop(columns=col1)

    elif product_type == "대입경":
        bad = df_y[df_y["STEP_SIZE_DMIN_01"].isna()].index
        df_x, df_y = df_x.drop(bad), df_y.drop(bad)

    return df_x, df_y
