import pytest
import pandas as pd
from src.data.preprocessor import DataPreprocesser
from src.data.tracker import TrackerRawData


def test_preprocessor_wide_format(dummy_excel_data, seq_1_6_path):
    tracker = TrackerRawData(dummy_excel_data, ["1라인"], "N86L")
    prep = DataPreprocesser(
        dict_raw=tracker.dict_lines_tracked,
        list_lines=["1라인"],
        seq_path=seq_1_6_path,
        is_material=False,
        is_handrecorded=True,
    )
    df = prep.dict_lines_preprocessed["1라인"]
    assert isinstance(df, pd.DataFrame)
    assert "lot_target" in df.columns
    assert "lot_reacted" in df.columns
    # 초기조건
    assert "반응투입_초기조건_pH" in df.columns
    # 스텝 피처
    assert "STEP_WEIGHT_Metal_01" in df.columns


def test_preprocessor_step_size_columns_exist(dummy_excel_data, seq_1_6_path):
    tracker = TrackerRawData(dummy_excel_data, ["1라인"], "N86L")
    prep = DataPreprocesser(
        dict_raw=tracker.dict_lines_tracked,
        list_lines=["1라인"],
        seq_path=seq_1_6_path,
        is_material=False,
        is_handrecorded=True,
    )
    df = prep.dict_lines_preprocessed["1라인"]
    size_cols = [c for c in df.columns if "STEP_SIZE_D50" in c]
    assert len(size_cols) > 0
