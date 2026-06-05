import pytest
import pandas as pd
import numpy as np
from src.model.postprocessor import split_xy, augment_data, filter_material


def make_dummy_df(n_rows=3, n_steps=5):
    data = {"lot_target": [f"LOT-{i}" for i in range(n_rows)],
            "lot_reacted": [f"REACT-{i}" for i in range(n_rows)],
            "반응투입_초기조건_pH": [11.0] * n_rows}
    for s in range(1, n_steps + 1):
        s_str = str(s).zfill(2)
        data[f"STEP_WEIGHT_Metal_{s_str}"] = [100.0] * n_rows
        data[f"STEP_SIZE_D50_{s_str}"] = [8.0 + s * 0.1] * n_rows
    return pd.DataFrame(data)


def test_split_xy_separates_size_columns():
    df = make_dummy_df()
    x, y = split_xy(df)
    assert all(c.startswith("STEP_SIZE_") for c in y.columns)
    assert not any(c.startswith("STEP_SIZE_") for c in x.columns)


def test_augment_produces_step_rows():
    df = make_dummy_df(n_rows=2, n_steps=5)
    df_aug = augment_data(df)
    # 배치 1개당 D50이 있는 스텝 수만큼 행 생성 = 2 * 5 = 10
    assert len(df_aug) == 10


def test_augment_masks_future_steps():
    df = make_dummy_df(n_rows=1, n_steps=5)
    df_aug = augment_data(df)
    # 2번째 행(j=1): STEP_WEIGHT_Metal_03 이후는 0.0이어야 함
    row_j1 = df_aug.iloc[1]
    assert row_j1["STEP_WEIGHT_Metal_01"] == 100.0
    assert row_j1["STEP_WEIGHT_Metal_02"] == 100.0
    assert row_j1["STEP_WEIGHT_Metal_03"] == 0.0
