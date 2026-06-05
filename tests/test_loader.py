import pytest
from src.data.loader import get_alldata


def test_get_alldata_returns_defaultdict(tmp_path):
    """실제 Excel 없이 빈 폴더면 빈 dict 반환해야 함."""
    result = get_alldata(str(tmp_path))
    assert isinstance(result, dict)


def test_get_alldata_sheet_count_dispatch(tmp_path):
    """1/2/3 sheet 파일 분류가 올바른지 확인."""
    from src.data.loader import TASKS_1SHEET, TASKS_2SHEET, TASKS_3SHEET
    assert "통합일지" in TASKS_1SHEET
    assert "수기운전일지" in TASKS_1SHEET
    assert "용해작업실적" in TASKS_2SHEET
    assert "반응투입스케쥴실적" in TASKS_3SHEET
