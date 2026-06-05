import pytest
from tests.conftest import (
    make_dummy_통합일지, make_dummy_수기운전일지,
    make_dummy_반응투입스케쥴실적, make_dummy_용해작업실적,
    make_dummy_원재료,
)
from src.data.tracker import TrackerRawData, extract_lot, remove_trailing_non_digits
from collections import defaultdict
import re


def test_extract_lot_parses_valid():
    lot = "N86L-1A250504-02"
    assert extract_lot(lot) == "N86L-1A250504-02"


def test_remove_trailing_non_digits():
    assert remove_trailing_non_digits("N86L-1A250504-02A") == "N86L-1A250504-02"
    assert remove_trailing_non_digits("N86L-1A250504-02") == "N86L-1A250504-02"


def test_tracker_returns_dict_per_line(dummy_excel_data):
    tracker = TrackerRawData(
        data_excel=dummy_excel_data,
        list_lines=["1라인"],
        product_name="N86L",
    )
    assert "1라인" in tracker.dict_lines_tracked
    df = tracker.dict_lines_tracked["1라인"]
    assert "lot_target" in df.columns
    assert "lot_reacted" in df.columns


def test_tracker_has_step_info_columns(dummy_excel_data):
    tracker = TrackerRawData(
        data_excel=dummy_excel_data,
        list_lines=["1라인"],
        product_name="N86L",
    )
    df = tracker.dict_lines_tracked["1라인"]
    # step_info_ 컬럼들이 있어야 함
    step_cols = [c for c in df.columns if c.startswith("step_info_")]
    assert len(step_cols) > 0
