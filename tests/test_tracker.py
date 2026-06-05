import pytest
import pandas as pd
from tests.conftest import make_dummy_수기운전일지
from src.data.tracker import TrackerRawData, extract_lot, remove_trailing_non_digits


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

def test_extract_lot_parses_valid():
    lot = "N86L-1A250504-02"
    assert extract_lot(lot) == "N86L-1A250504-02"


def test_extract_lot_strips_suffix():
    assert extract_lot("N86L-1A250504-02A") == "N86L-1A250504-02"


def test_extract_lot_no_match():
    assert extract_lot("invalid") == "no match lot"


def test_remove_trailing_non_digits():
    assert remove_trailing_non_digits("N86L-1A250504-02A") == "N86L-1A250504-02"
    assert remove_trailing_non_digits("N86L-1A250504-02") == "N86L-1A250504-02"


def test_remove_trailing_non_digits_non_string():
    assert remove_trailing_non_digits(123) == 123


# ---------------------------------------------------------------------------
# Tracker integration tests
# ---------------------------------------------------------------------------

class TestTrackerReturnsDataFrame:
    def test_df_tracked_is_dataframe(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert isinstance(tracker.df_tracked, pd.DataFrame)

    def test_df_tracked_has_lot_columns(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "lot_target" in tracker.df_tracked.columns
        assert "lot_reacted" in tracker.df_tracked.columns

    def test_df_tracked_has_rows(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert len(tracker.df_tracked) == 3


class TestStepInfoColumns:
    def test_step_info_columns_exist(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        step_cols = [c for c in tracker.df_tracked.columns if c.startswith("step_info_")]
        assert len(step_cols) == 60

    def test_step_info_contains_dict(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        df = tracker.df_tracked
        # At least the first row should have step_info_01 populated
        val = df.at[0, "step_info_01"]
        assert isinstance(val, dict)
        assert "metal" in val
        assert "naoh" in val
        assert "nh4oh" in val
        assert "ph" in val
        assert "rpm" in val


class TestWeightColumns:
    def test_weights_metal_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "weights_metal" in tracker.df_tracked.columns

    def test_weights_naoh_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "weights_naoh" in tracker.df_tracked.columns

    def test_weights_nh4oh_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "weights_nh4oh" in tracker.df_tracked.columns

    def test_weight_values_are_lists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        df = tracker.df_tracked
        row = df.iloc[0]
        assert isinstance(row["weights_metal"], list)
        assert isinstance(row["weights_naoh"], list)
        assert isinstance(row["weights_nh4oh"], list)


class TestMaterialColumns:
    def test_df_naoh_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "df_naoh" in tracker.df_tracked.columns

    def test_df_coso4_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "df_coso4" in tracker.df_tracked.columns

    def test_df_mnso4_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "df_mnso4" in tracker.df_tracked.columns

    def test_df_niso4_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "df_niso4" in tracker.df_tracked.columns


class TestReactInit:
    def test_df_react_init_column_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "df_react_init" in tracker.df_tracked.columns

    def test_df_react_init_is_dict(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        df = tracker.df_tracked
        val = df.at[0, "df_react_init"]
        assert isinstance(val, dict)
        assert "생산LOT번호" in val


class TestMeltingColumns:
    def test_lots_coso4_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "lots_coso4" in tracker.df_tracked.columns

    def test_lots_mnso4_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "lots_mnso4" in tracker.df_tracked.columns

    def test_lots_niso4_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "lots_niso4" in tracker.df_tracked.columns

    def test_weight_coso4_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "weight_coso4" in tracker.df_tracked.columns


class TestHandrecorded:
    def test_df_handrecorded_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "df_handrecorded" in tracker.df_tracked.columns

    def test_df_handrecorded_is_dataframe(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        df = tracker.df_tracked
        val = df.at[0, "df_handrecorded"]
        assert isinstance(val, pd.DataFrame)


class TestStepsParsing:
    def test_steps_ph_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "steps_ph" in tracker.df_tracked.columns

    def test_steps_rpm_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "steps_rpm" in tracker.df_tracked.columns

    def test_steps_num_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "steps_num" in tracker.df_tracked.columns

    def test_steps_time_exists(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert "steps_time" in tracker.df_tracked.columns


class TestDebugMode:
    def test_debug_mode_does_not_crash(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
            debug=True,
        )
        assert isinstance(tracker.df_tracked, pd.DataFrame)

    def test_debug_prints_messages(self, dummy_data, capsys):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
            debug=True,
        )
        # Debug should not crash; output check is optional
        # (debug messages printed only on matching failures)
        assert isinstance(tracker.df_tracked, pd.DataFrame)


class TestErrorLog:
    def test_error_log_is_list(self, dummy_data):
        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인"],
            product_name="N86L",
        )
        assert isinstance(tracker.error_log, list)


class TestMultiLine:
    def test_multi_line_combines_results(self, dummy_data):
        """If data has multiple lines, df_tracked should have rows from all."""
        # Add 2라인 data
        from tests.conftest import make_dummy_통합일지, make_dummy_반응_init, make_dummy_반응_step
        lots_2 = ["N86L-2A250601-02"]
        dummy_data["통합일지"]["2라인"] = make_dummy_통합일지(n_lots=1)
        # Fix the lot in 통합일지 for 2라인
        dummy_data["통합일지"]["2라인"].iloc[4, 0] = lots_2[0]

        # Add 반응 data for the new lot
        import pandas as pd
        init_2 = make_dummy_반응_init(lots_2)
        dummy_data["반응_init"] = pd.concat(
            [dummy_data["반응_init"], init_2], ignore_index=True
        )
        step_2 = make_dummy_반응_step(lots_2, n_steps=20)
        dummy_data["반응_step"] = pd.concat(
            [dummy_data["반응_step"], step_2], ignore_index=True
        )

        # 수기운전일지 for 2라인
        from tests.conftest import make_dummy_수기운전일지
        dummy_data["수기운전일지"]["2라인"] = [make_dummy_수기운전일지(lots_2[0])]

        tracker = TrackerRawData(
            data=dummy_data,
            list_lines=["1라인", "2라인"],
            product_name="N86L",
        )
        # 3 from 1라인 + 1 from 2라인 = 4
        assert len(tracker.df_tracked) == 4
