import typing as tp
import pandas as pd
import numpy as np

INT_LENGTH = 60

INIT_COLUMNS = [
    "반응투입_초기조건_작업시간",
    "반응투입_초기조건_순수온도",
    "반응투입_초기조건_순수투입중량(kg)",
    "반응투입_초기조건_NAOH투입중량(kg)",
    "반응투입_초기조건_NH4OH투입중량(kg)",
    "반응투입_초기조건_용존산소량",
    "반응투입_초기조건_pH",
    "반응투입_초기조건_N2주입유량",
    "반응투입_초기조건_N2PURGE시간",
    "반응투입_초기조건_IMPELLAR교반속도",
]

SHEET3_COLS = {
    "반응투입_초기조건_작업시간": "작업\n시간",
    "반응투입_초기조건_순수온도": "순수\n온도",
    "반응투입_초기조건_순수투입중량(kg)": "순수\n투입 중량(kg)",
    "반응투입_초기조건_NAOH투입중량(kg)": "NAOH\n투입중량(kg)",
    "반응투입_초기조건_NH4OH투입중량(kg)": "NH4OH\n투입중량(kg)",
    "반응투입_초기조건_용존산소량": "용존산소량",
    "반응투입_초기조건_pH": "pH",
    "반응투입_초기조건_N2주입유량": "N2\n주입 유량",
    "반응투입_초기조건_N2PURGE시간": "N2\nPURGE 시간",
    "반응투입_초기조건_IMPELLAR교반속도": "IMPELLAR\n교반 속도",
}

# 수기운전일지 컬럼 인덱스 (time, dmin, d10, d50, d90, dmax)
HAND_COL_INDICES = [0, 2, 3, 4, 5, 6]
HAND_COL_NAMES = ["time", "dmin", "d10", "d50", "d90", "dmax"]

_STEP_PREFIXES = ["STEP_WEIGHT_Metal_", "STEP_WEIGHT_NaOH_", "STEP_WEIGHT_NH4OH_",
                  "STEP_PH_", "STEP_RPM_"]
_SIZE_PREFIXES = ["STEP_SIZE_DMIN_", "STEP_SIZE_D10_", "STEP_SIZE_D50_",
                  "STEP_SIZE_D90_", "STEP_SIZE_DMAX_"]


def _load_seq(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["idx", "time"])


def _step_col(prefix: str, j: int) -> str:
    return f"{prefix}{str(j + 1).zfill(2)}"


class DataPreprocesser:

    def __init__(
        self,
        dict_raw: tp.Dict[str, pd.DataFrame],
        list_lines: tp.List[str],
        seq_path: str,
        is_material: bool,
        is_handrecorded: bool,
    ) -> None:
        self.dict_raw = dict_raw
        self.list_lines = list_lines
        self.is_material = is_material
        self.is_handrecorded = is_handrecorded
        self.df_seq = _load_seq(seq_path)
        self.dict_lines_preprocessed, self.dict_df_filtered = self._preprocess_all()

    def _preprocess_all(self) -> tp.Tuple[dict, dict]:
        preprocessed, filtered = {}, {}
        for line in self.list_lines:
            df_raw = self.dict_raw[line]
            df_raw = self._filter(df_raw)
            df_pre = self._build_features(df_raw)
            preprocessed[line] = df_pre
            filtered[line] = df_raw
        return preprocessed, filtered

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_material:
            non_step_cols = [c for c in df.columns
                             if not any(t in c for t in ["df_handrecorded", "step_info", "df_coso4"])]
            df = df.dropna(subset=non_step_cols)
        else:
            df = df.dropna(subset=["df_reacted"])

        if self.is_handrecorded:
            df = df.dropna(subset=["df_handrecorded", "steps_num"])

        return df.reset_index(drop=True)

    def _build_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        n = len(df_raw)
        idx = df_raw.index

        # 초기조건: pre-allocate as float64 (np.nan seeds float dtype, not object)
        init_data = {col: np.full(n, np.nan) for col in INIT_COLUMNS}
        for i, (_, row) in enumerate(df_raw.iterrows()):
            sheet3 = row["df_reacted"]["Sheet3"]
            for feat_col, sheet_col in SHEET3_COLS.items():
                val = str(sheet3[sheet_col][0]).replace(",", "")
                init_data[feat_col][i] = float(val)

        # 스텝 피처: pre-allocate as float64
        step_data: tp.Dict[str, np.ndarray] = {
            _step_col(prefix, j): np.full(n, np.nan)
            for prefix in _STEP_PREFIXES
            for j in range(INT_LENGTH)
        }
        for i, (_, row) in enumerate(df_raw.iterrows()):
            wm = row["weights_metal"]
            wn = row["weights_naoh"]
            wnh = row["weights_nh4oh"]
            rpm = row["steps_rpm"]
            ph = row["steps_ph"]
            for j in range(INT_LENGTH):
                if wm is not None and j < len(wm):
                    step_data[_step_col("STEP_WEIGHT_Metal_", j)][i] = wm[j]
                if wn is not None and j < len(wn):
                    step_data[_step_col("STEP_WEIGHT_NaOH_", j)][i] = wn[j]
                if wnh is not None and j < len(wnh):
                    step_data[_step_col("STEP_WEIGHT_NH4OH_", j)][i] = wnh[j]
                if rpm is not None and j < len(rpm):
                    step_data[_step_col("STEP_RPM_", j)][i] = rpm[j]
                if ph is not None and j < len(ph):
                    step_data[_step_col("STEP_PH_", j)][i] = ph[j]

        # 한 번에 concat — fragmentation 방지
        df = pd.concat([
            df_raw[["lot_target", "lot_reacted"]].reset_index(drop=True),
            pd.DataFrame(init_data),
            pd.DataFrame(step_data),
        ], axis=1)

        if self.is_handrecorded:
            df = self._interpolate_handrecorded(df, df_raw)

        return df

    def _interpolate_handrecorded(
        self, df: pd.DataFrame, df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """수기운전일지 D50 값을 스텝 누적시간으로 선형 보간."""
        n = len(df_raw)
        cumtimes = np.cumsum(self.df_seq["time"].tolist())

        # pre-allocate size columns
        size_data: tp.Dict[str, np.ndarray] = {
            _step_col(prefix, j): np.full(n, np.nan)
            for prefix in _SIZE_PREFIXES
            for j in range(INT_LENGTH)
        }

        col_map = {"dmin": "STEP_SIZE_DMIN_", "d10": "STEP_SIZE_D10_",
                   "d50": "STEP_SIZE_D50_", "d90": "STEP_SIZE_D90_",
                   "dmax": "STEP_SIZE_DMAX_"}

        for i, (_, row) in enumerate(df_raw.iterrows()):
            df_hand = row["df_handrecorded"]
            if df_hand is None or (isinstance(df_hand, float) and np.isnan(df_hand)):
                continue

            len_steps = len(row["steps_num"] or [])
            cols = df_hand.columns
            target_cols = [cols[k] for k in HAND_COL_INDICES]
            df_h = df_hand.iloc[12:][target_cols].copy()
            df_h.columns = HAND_COL_NAMES

            last_d50 = df_h["d50"].last_valid_index()
            last_time = df_h["time"].last_valid_index()
            if last_d50 is None or last_time is None:
                continue
            df_h = df_h.loc[:min(last_d50, last_time)]
            df_h["time"] = df_h["time"].astype(str).str.replace(",", "").astype(int)

            time_max = df_h["time"].max()
            for col in HAND_COL_NAMES[1:]:
                df_h[col] = pd.to_numeric(df_h[col], errors="coerce")

            df_indexed = df_h.set_index("time").dropna(how="all")
            all_times = sorted(set(df_indexed.index.tolist() + cumtimes.tolist()))
            df_interp = df_indexed.reindex(all_times).interpolate(method="linear")
            df_interp.loc[df_interp.index > time_max] = np.nan
            df_at_steps = df_interp.loc[cumtimes].reset_index()

            for j in range(INT_LENGTH):
                if j >= min(len(df_at_steps), len_steps):
                    break
                for hand_col, prefix in col_map.items():
                    size_data[_step_col(prefix, j)][i] = df_at_steps[hand_col].iloc[j]

        df = pd.concat([df, pd.DataFrame(size_data, index=df.index)], axis=1)
        return df
