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
]

INIT_COL_MAP = {
    "반응투입_초기조건_작업시간": "작업시간",
    "반응투입_초기조건_순수온도": "순수온도",
    "반응투입_초기조건_순수투입중량(kg)": "순수투입중량(kg)",
    "반응투입_초기조건_NAOH투입중량(kg)": "NAOH순수투입중량(kg)",
    "반응투입_초기조건_NH4OH투입중량(kg)": "NH4OH순수투입중량(kg)",
    "반응투입_초기조건_용존산소량": "용존산소량",
    "반응투입_초기조건_pH": "Ph",
    "반응투입_초기조건_N2주입유량": "N2주입유량",
    "반응투입_초기조건_N2PURGE시간": "N2 PURGE시간",
}

# 수기운전일지 컬럼 인덱스 (time, dmin, d10, d50, d90, dmax)
HAND_COL_INDICES = [0, 2, 3, 4, 5, 6]
HAND_COL_NAMES = ["time", "dmin", "d10", "d50", "d90", "dmax"]

_STEP_PREFIXES = ["STEP_WEIGHT_Metal_", "STEP_WEIGHT_NaOH_", "STEP_WEIGHT_NH4OH_",
                  "STEP_PH_", "STEP_RPM_"]
_SIZE_PREFIXES = ["STEP_SIZE_DMIN_", "STEP_SIZE_D10_", "STEP_SIZE_D50_",
                  "STEP_SIZE_D90_", "STEP_SIZE_DMAX_"]

# Material weighted-sum feature suffixes (must align with postprocessor MATERIAL_FILTER_TERMS)
_MATERIAL_SUFFIXES = [
    "_황산망간_성분_ChemicalComposition(C01)_Mn",
    "_황산망간_성분_InitialPH(C03)_pH",
    "_황산니켈_성분_ChemicalComposition(C01)_ni",
    "_가성소다_성분_ChemicalComposition(C01)_NaOH",
    "_황산코발트_투입량",
]


def _load_seq(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["idx", "time"])


def _step_col(prefix: str, j: int) -> str:
    return f"{prefix}{str(j + 1).zfill(2)}"


def _case_insensitive_get(d: dict, key: str) -> tp.Any:
    """Lookup key in dict d with case-insensitive matching."""
    key_lower = key.lower()
    for k, v in d.items():
        if k.lower() == key_lower:
            return v
    return None


class DataPreprocesser:

    def __init__(
        self,
        df_tracked: pd.DataFrame,
        seq_path: str,
        is_material: bool,
        is_handrecorded: bool,
    ) -> None:
        self.df_tracked = df_tracked
        self.is_material = is_material
        self.is_handrecorded = is_handrecorded
        self.df_seq = _load_seq(seq_path)
        self.df_preprocessed, self.df_filtered = self._preprocess()

    def _preprocess(self) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = self._filter(self.df_tracked)
        df_pre = self._build_features(df_raw)
        return df_pre, df_raw

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_material:
            non_step_cols = [c for c in df.columns
                             if not any(t in c for t in ["df_handrecorded", "step_info", "df_coso4"])]
            df = df.dropna(subset=non_step_cols)
        else:
            df = df.dropna(subset=["df_react_init"])

        if self.is_handrecorded:
            df = df.dropna(subset=["df_handrecorded", "steps_num"])

        return df.reset_index(drop=True)

    def _build_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        n = len(df_raw)

        # 초기조건: 9 columns from df_react_init dict (case-insensitive)
        init_data = {col: np.full(n, np.nan) for col in INIT_COLUMNS}
        for i, (_, row) in enumerate(df_raw.iterrows()):
            react_init = row["df_react_init"]
            if react_init is None or not isinstance(react_init, dict):
                continue
            for feat_col, source_key in INIT_COL_MAP.items():
                val = _case_insensitive_get(react_init, source_key)
                if val is not None:
                    try:
                        init_data[feat_col][i] = float(str(val).replace(",", ""))
                    except (ValueError, TypeError):
                        pass

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

        if self.is_material:
            df = self._build_material_features(df, df_raw)

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

    def _build_material_features(
        self, df: pd.DataFrame, df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """Build STEP_MATERIAL_ weighted-sum features for each step.

        For each batch row, at each step j (1..60), compute:
          1. MnSO4 Mn weighted average: cumulative MnSO4 weights * property / total weight
          2. MnSO4 pH weighted average
          3. NiSO4 Ni weighted average
          4. NaOH NaOH weighted average
          5. CoSO4 total cumulative weight (not weighted average)
        """
        n = len(df_raw)
        # Pre-allocate all material columns
        mat_data: tp.Dict[str, np.ndarray] = {}
        for j in range(INT_LENGTH):
            step_num = str(j + 1).zfill(2)
            for suffix in _MATERIAL_SUFFIXES:
                col_name = f"STEP_MATERIAL_{step_num}{suffix}"
                mat_data[col_name] = np.full(n, np.nan)

        for i, (_, row) in enumerate(df_raw.iterrows()):
            # Get material info lists (from tracker)
            df_mnso4_info = row.get("df_mnso4")   # list of lists of lists
            df_niso4_info = row.get("df_niso4")   # list of lists of lists
            df_naoh_info = row.get("df_naoh")      # list of lists
            df_coso4_info = row.get("df_coso4")    # list of lists of lists

            weight_coso4 = row.get("weight_coso4")  # list of lists of weights
            weight_mnso4 = row.get("weight_mnso4")  # list of lists of weights
            weight_niso4 = row.get("weight_niso4")  # list of lists of weights

            for j in range(INT_LENGTH):
                step_num = str(j + 1).zfill(2)
                step_info_col = f"step_info_{step_num}"
                step_info = row.get(step_info_col)
                if step_info is None or not isinstance(step_info, dict):
                    continue

                # --- MnSO4 Mn weighted average ---
                mnso4_mn = self._compute_metal_weighted_avg(
                    step_info, "metal", df_mnso4_info, weight_mnso4, prop_idx=1
                )
                if mnso4_mn is not None:
                    mat_data[f"STEP_MATERIAL_{step_num}_황산망간_성분_ChemicalComposition(C01)_Mn"][i] = mnso4_mn

                # --- MnSO4 pH weighted average ---
                mnso4_ph = self._compute_metal_weighted_avg(
                    step_info, "metal", df_mnso4_info, weight_mnso4, prop_idx=2
                )
                if mnso4_ph is not None:
                    mat_data[f"STEP_MATERIAL_{step_num}_황산망간_성분_InitialPH(C03)_pH"][i] = mnso4_ph

                # --- NiSO4 Ni weighted average ---
                niso4_ni = self._compute_metal_weighted_avg(
                    step_info, "metal", df_niso4_info, weight_niso4, prop_idx=1
                )
                if niso4_ni is not None:
                    mat_data[f"STEP_MATERIAL_{step_num}_황산니켈_성분_ChemicalComposition(C01)_ni"][i] = niso4_ni

                # --- NaOH NaOH weighted average ---
                naoh_val = self._compute_naoh_weighted_avg(
                    step_info, df_naoh_info
                )
                if naoh_val is not None:
                    mat_data[f"STEP_MATERIAL_{step_num}_가성소다_성분_ChemicalComposition(C01)_NaOH"][i] = naoh_val

                # --- CoSO4 total cumulative weight ---
                coso4_total = self._compute_coso4_total_weight(
                    step_info, weight_coso4
                )
                if coso4_total is not None:
                    mat_data[f"STEP_MATERIAL_{step_num}_황산코발트_투입량"][i] = coso4_total

        df = pd.concat([df, pd.DataFrame(mat_data, index=df.index)], axis=1)
        return df

    @staticmethod
    def _compute_metal_weighted_avg(
        step_info: dict,
        metal_key: str,
        df_material_info: tp.Optional[list],
        weight_groups: tp.Optional[list],
        prop_idx: int,
    ) -> tp.Optional[float]:
        """Compute weighted average of a material property using metal lot weights.

        The metal lot list in step_info tells us which lots contributed so far.
        weight_groups (from tracker melting) is list-of-lists: one sublist per metal LOT.
        df_material_info is list-of-lists-of-lists: one group per metal LOT,
        each containing sublists of [lot_no, prop1, prop2, ...].

        We compute: sum(cumulative_weight_k * property_k) / sum(cumulative_weight_k)
        where k iterates over the metal LOTs that have contributed up to this step.
        """
        if df_material_info is None or weight_groups is None:
            return None

        metal_info = step_info.get(metal_key)
        if metal_info is None:
            return None

        lot_list = metal_info.get("lot_metal", [])
        weight_dict = metal_info.get("weight", {})

        if not lot_list:
            return None

        total_weighted = 0.0
        total_weight = 0.0

        for k, lot_m in enumerate(lot_list):
            # Cumulative weight for this metal LOT
            lot_weights = weight_dict.get(lot_m, [])
            cum_w = sum(float(w) for w in lot_weights)
            if cum_w == 0:
                continue

            # Get property value from df_material_info
            # df_material_info is grouped by metal LOT (same index as lot_list order
            # in lots_metal). Each group contains sublists for each raw material lot.
            if k >= len(df_material_info):
                continue

            group = df_material_info[k]
            # group is a list of lists: [[lot_no, prop1, ...], ...]
            # For the weighted average, we average over all raw material lots
            # in this group equally (they map to the same metal LOT).
            if not group:
                continue

            # Simple approach: average property across all raw lots in group,
            # then weight by cumulative metal weight
            prop_sum = 0.0
            prop_count = 0
            for mat_row in group:
                if prop_idx < len(mat_row):
                    try:
                        prop_sum += float(mat_row[prop_idx])
                        prop_count += 1
                    except (ValueError, TypeError):
                        pass
            if prop_count == 0:
                continue

            avg_prop = prop_sum / prop_count
            total_weighted += cum_w * avg_prop
            total_weight += cum_w

        if total_weight == 0:
            return None
        return total_weighted / total_weight

    @staticmethod
    def _compute_naoh_weighted_avg(
        step_info: dict,
        df_naoh_info: tp.Optional[list],
    ) -> tp.Optional[float]:
        """Compute NaOH weighted average using step_info naoh cumulative weights.

        df_naoh_info is a list of lists: [[lot_no, naoh_value], ...]
        step_info["naoh"]["lot_naoh"] = [lot1, lot2, ...]
        step_info["naoh"]["weight"] = {lot1: [w1, w2, ...], lot2: [...]}
        """
        if df_naoh_info is None:
            return None

        naoh_info = step_info.get("naoh")
        if naoh_info is None:
            return None

        lot_list = naoh_info.get("lot_naoh", [])
        weight_dict = naoh_info.get("weight", {})

        if not lot_list:
            return None

        # Build lot -> property map from df_naoh_info
        # df_naoh_info: [[lot_no, naoh_value], ...]
        lot_prop_map = {}
        for info_row in df_naoh_info:
            if len(info_row) >= 2:
                lot_no = str(info_row[0])
                try:
                    prop_val = float(info_row[1])
                    lot_prop_map[lot_no] = prop_val
                except (ValueError, TypeError):
                    pass

        total_weighted = 0.0
        total_weight = 0.0

        for lot_n in lot_list:
            lot_weights = weight_dict.get(lot_n, [])
            cum_w = sum(float(w) for w in lot_weights)
            if cum_w == 0:
                continue

            prop_val = lot_prop_map.get(lot_n)
            if prop_val is None:
                continue

            total_weighted += cum_w * prop_val
            total_weight += cum_w

        if total_weight == 0:
            return None
        return total_weighted / total_weight

    @staticmethod
    def _compute_coso4_total_weight(
        step_info: dict,
        weight_coso4: tp.Optional[list],
    ) -> tp.Optional[float]:
        """Compute total cumulative CoSO4 weight at this step.

        weight_coso4 is list of lists of weights (grouped by metal LOT).
        We sum all weights across all groups.

        We use the metal lot list from step_info to know how many metal LOTs
        have contributed so far, then sum CoSO4 weights for those groups.
        """
        if weight_coso4 is None:
            return None

        metal_info = step_info.get("metal")
        if metal_info is None:
            return None

        lot_list = metal_info.get("lot_metal", [])
        n_lots = len(lot_list)

        total = 0.0
        for k in range(min(n_lots, len(weight_coso4))):
            group = weight_coso4[k]
            for w in group:
                total += float(w)

        return total if total > 0 else None
