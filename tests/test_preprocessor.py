import pytest
import numpy as np
import pandas as pd

from src.data.tracker import TrackerRawData
from src.data.preprocessor import DataPreprocesser, INIT_COLUMNS, INIT_COL_MAP


# ---------------------------------------------------------------------------
# Helper: build tracker output from dummy_data fixture
# ---------------------------------------------------------------------------

def _make_tracked(dummy_data):
    tracker = TrackerRawData(
        data=dummy_data,
        list_lines=["1라인"],
        product_name="N86L",
    )
    return tracker.df_tracked


# ---------------------------------------------------------------------------
# Constructor API
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_accepts_single_dataframe(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        assert isinstance(prep.df_preprocessed, pd.DataFrame)
        assert isinstance(prep.df_filtered, pd.DataFrame)

    def test_no_list_lines_or_dict_raw_params(self, dummy_data, seq_1_6_path):
        """Constructor should NOT accept old-style dict_raw/list_lines."""
        df_tracked = _make_tracked(dummy_data)
        with pytest.raises(TypeError):
            DataPreprocesser(
                dict_raw={"1라인": df_tracked},
                list_lines=["1라인"],
                seq_path=seq_1_6_path,
                is_material=False,
                is_handrecorded=False,
            )


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

class TestInitialConditions:
    def test_no_impellar_column(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        impellar_cols = [c for c in df.columns if "IMPELLAR" in c.upper()]
        assert len(impellar_cols) == 0

    def test_nine_init_columns(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        init_cols = [c for c in df.columns if c.startswith("반응투입_초기조건_")]
        assert len(init_cols) == 9

    def test_init_columns_match_spec(self, dummy_data, seq_1_6_path):
        """All 9 specified INIT_COLUMNS should be present."""
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        for col in INIT_COLUMNS:
            assert col in df.columns, f"Missing: {col}"

    def test_init_values_are_numeric(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        for col in INIT_COLUMNS:
            assert df[col].dtype in [np.float64, np.int64, float, int], (
                f"Column {col} is {df[col].dtype}"
            )

    def test_init_col_map_has_nine_entries(self):
        assert len(INIT_COL_MAP) == 9
        assert len(INIT_COLUMNS) == 9

    def test_case_insensitive_matching(self, dummy_data, seq_1_6_path):
        """Even if df_react_init keys have different casing, values should be found."""
        df_tracked = _make_tracked(dummy_data)
        # Modify one row's df_react_init to have a differently-cased key
        init_dict = df_tracked.at[0, "df_react_init"].copy()
        # Change "Ph" to "ph" — should still match
        if "Ph" in init_dict:
            init_dict["ph"] = init_dict.pop("Ph")
            df_tracked.at[0, "df_react_init"] = init_dict
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        # pH column should still have a valid value for row 0
        assert not np.isnan(df.at[0, "반응투입_초기조건_pH"])


# ---------------------------------------------------------------------------
# Step features (STEP_WEIGHT, STEP_PH, STEP_RPM)
# ---------------------------------------------------------------------------

class TestStepFeatures:
    def test_step_weight_metal_columns(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        wm_cols = [c for c in df.columns if c.startswith("STEP_WEIGHT_Metal_")]
        assert len(wm_cols) == 60

    def test_step_weight_naoh_columns(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        wn_cols = [c for c in df.columns if c.startswith("STEP_WEIGHT_NaOH_")]
        assert len(wn_cols) == 60

    def test_step_weight_nh4oh_columns(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        wnh_cols = [c for c in df.columns if c.startswith("STEP_WEIGHT_NH4OH_")]
        assert len(wnh_cols) == 60

    def test_step_ph_columns(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        ph_cols = [c for c in df.columns if c.startswith("STEP_PH_")]
        assert len(ph_cols) == 60

    def test_step_rpm_columns(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        rpm_cols = [c for c in df.columns if c.startswith("STEP_RPM_")]
        assert len(rpm_cols) == 60

    def test_step_weight_values_populated(self, dummy_data, seq_1_6_path):
        """First 20 steps should have non-NaN weight values."""
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        for j in range(1, 21):
            col = f"STEP_WEIGHT_Metal_{str(j).zfill(2)}"
            assert not df[col].isna().all(), f"{col} is all NaN"


# ---------------------------------------------------------------------------
# Handrecorded columns (STEP_SIZE_)
# ---------------------------------------------------------------------------

class TestHandrecorded:
    def test_step_size_columns_exist(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=True,
        )
        df = prep.df_preprocessed
        size_cols = [c for c in df.columns if c.startswith("STEP_SIZE_")]
        assert len(size_cols) > 0

    def test_step_size_d50_columns(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=True,
        )
        df = prep.df_preprocessed
        d50_cols = [c for c in df.columns if "STEP_SIZE_D50" in c]
        assert len(d50_cols) == 60

    def test_no_size_columns_when_disabled(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        size_cols = [c for c in df.columns if c.startswith("STEP_SIZE_")]
        assert len(size_cols) == 0


# ---------------------------------------------------------------------------
# Material weighted sum (STEP_MATERIAL_ columns)
# ---------------------------------------------------------------------------

class TestMaterialWeightedSum:
    def test_step_material_columns_exist(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=True,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        mat_cols = [c for c in df.columns if c.startswith("STEP_MATERIAL_")]
        assert len(mat_cols) > 0

    def test_no_material_columns_when_disabled(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        mat_cols = [c for c in df.columns if c.startswith("STEP_MATERIAL_")]
        assert len(mat_cols) == 0

    def test_material_column_naming_pattern(self, dummy_data, seq_1_6_path):
        """Columns must match STEP_MATERIAL_{NN}_{suffix} pattern."""
        import re
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=True,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        mat_cols = [c for c in df.columns if c.startswith("STEP_MATERIAL_")]
        pattern = re.compile(r"^STEP_MATERIAL_(\d{2})_")
        for col in mat_cols:
            assert pattern.match(col), f"Column {col} does not match pattern"

    def test_material_columns_match_postprocessor_pattern(self, dummy_data, seq_1_6_path):
        """Postprocessor expects PATTERN_MATERIAL = r'^STEP_MATERIAL_(\\d+)'."""
        import re
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=True,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        pattern = re.compile(r"^STEP_MATERIAL_(\d+)")
        mat_cols = [c for c in df.columns if c.startswith("STEP_MATERIAL_")]
        for col in mat_cols:
            m = pattern.match(col)
            assert m is not None, f"Column {col} doesn't match postprocessor pattern"
            step_num = int(m.group(1))
            assert 1 <= step_num <= 60

    def test_five_material_suffixes_per_step(self, dummy_data, seq_1_6_path):
        """Each step should have 5 material feature columns."""
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=True,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        # Check step 01 has 5 material columns
        step01_mat = [c for c in df.columns if c.startswith("STEP_MATERIAL_01_")]
        assert len(step01_mat) == 5

    def test_material_total_columns_count(self, dummy_data, seq_1_6_path):
        """60 steps * 5 suffixes = 300 STEP_MATERIAL_ columns."""
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=True,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        mat_cols = [c for c in df.columns if c.startswith("STEP_MATERIAL_")]
        assert len(mat_cols) == 300

    def test_material_suffixes_match_filter_terms(self, dummy_data, seq_1_6_path):
        """Column suffixes should match MATERIAL_FILTER_TERMS from postprocessor."""
        from src.model.postprocessor import MATERIAL_FILTER_TERMS
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=True,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        step01_mat = [c for c in df.columns if c.startswith("STEP_MATERIAL_01")]
        for term in MATERIAL_FILTER_TERMS:
            expected = f"STEP_MATERIAL_01{term}"
            assert expected in step01_mat, f"Missing: {expected}"


# ---------------------------------------------------------------------------
# Filter behavior
# ---------------------------------------------------------------------------

class TestFilter:
    def test_filter_uses_df_react_init(self, dummy_data, seq_1_6_path):
        """Rows with missing df_react_init should be filtered out."""
        df_tracked = _make_tracked(dummy_data)
        # Set one row's df_react_init to None
        df_tracked.at[0, "df_react_init"] = None
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        # Should have one fewer row
        assert len(df) == len(df_tracked) - 1


# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------

class TestOutputColumns:
    def test_has_lot_columns(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=False,
            is_handrecorded=False,
        )
        df = prep.df_preprocessed
        assert "lot_target" in df.columns
        assert "lot_reacted" in df.columns

    def test_output_is_dataframe(self, dummy_data, seq_1_6_path):
        df_tracked = _make_tracked(dummy_data)
        prep = DataPreprocesser(
            df_tracked=df_tracked,
            seq_path=seq_1_6_path,
            is_material=True,
            is_handrecorded=True,
        )
        assert isinstance(prep.df_preprocessed, pd.DataFrame)
        assert isinstance(prep.df_filtered, pd.DataFrame)
