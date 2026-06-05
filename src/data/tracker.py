import re
import copy
import typing as tp
import pandas as pd
import numpy as np

INT_LENGTH = 60

PATTERN_LOT = r"([A-Za-z]\d{2}[A-Za-z]-\d{1,2}[A-Za-z]\d{6}-\d{1,2})[A-Za-z]?.*"


def extract_lot(text: str) -> str:
    matches = re.findall(PATTERN_LOT, str(text))
    for match in matches:
        if len(match) >= 8 and any(c.isalpha() for c in match):
            return match.upper()
    return "no match lot"


def remove_trailing_non_digits(lot: str) -> str:
    if not isinstance(lot, str):
        return lot
    return re.sub(r"[^0-9]+$", "", lot)


def _debug(msg: str, debug: bool) -> None:
    if debug:
        print(f"[DEBUG] tracker: {msg}")


class TrackerRawData:
    """N86L/N86S LOT tracker for consolidated data format."""

    def __init__(
        self,
        data: dict,
        list_lines: tp.List[str],
        product_name: str,
        debug: bool = False,
    ) -> None:
        self.data = data
        self.list_lines = list_lines
        self.product_name = product_name
        self.debug = debug
        self.error_log: tp.List[str] = []
        self.df_tracked = self._track()

    def _track(self) -> pd.DataFrame:
        """Track all lines and return a single combined DataFrame."""
        frames = []
        for line in self.list_lines:
            df_integrated = self.data["통합일지"].get(line)
            if df_integrated is None:
                _debug(f"통합일지 없음: {line}", self.debug)
                continue
            df_line = self._track_line(df_integrated, line)
            frames.append(df_line)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _track_line(self, df_integrated: pd.DataFrame, line: str) -> pd.DataFrame:
        df_lots = self._extract_lot_pairs(df_integrated, line)
        df = self._attach_reaction(df_lots, line)
        df = self._attach_melting(df, line)
        df = self._attach_naoh(df, line)
        df = self._attach_materials(df, line)
        return df

    def _extract_lot_pairs(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
        """통합일지에서 (lot_reacted, lot_target) 쌍 추출."""
        has_no_col = "No." in df.columns

        if has_no_col:
            col_react = "No."
            col_target = "포장 Lot No" if "포장 Lot No" in df.columns else "Unnamed: 1"
        else:
            col_react = "생산품 구분\n(양산/DOE)"
            col_target = "포장 Lot No"

        df_lots = df[[col_react, col_target]].copy()
        df_lots.columns = ["lot_reacted", "lot_target"]
        df_lots = df_lots.loc[4:].reset_index(drop=True)
        df_lots["lot_reacted"] = df_lots["lot_reacted"].str.lstrip(" ").ffill()

        # Exclude mixed batches (2+ product names)
        lower = df_lots["lot_reacted"].str.lower()
        mixed = lower[lower.str.count(self.product_name.lower()) >= 2].index
        df_lots = df_lots.drop(mixed)

        df_lots = df_lots.dropna().reset_index(drop=True)
        df_lots = df_lots[df_lots["lot_target"].str.endswith("01")]
        df_lots["lot_reacted"] = df_lots["lot_reacted"].apply(extract_lot)
        return df_lots.reset_index(drop=True)

    def _attach_reaction(self, df_lots: pd.DataFrame, line: str) -> pd.DataFrame:
        """Attach 반응투입 (init + step) and 수기운전일지 data by LOT matching."""
        df = df_lots.copy()

        # Initialize columns
        new_cols = [
            "df_react_init", "df_handrecorded",
            "lots_metal", "lots_naoh", "lots_nh4oh",
            "weights_metal", "weights_naoh", "weights_nh4oh",
            "steps_num", "steps_time", "steps_rpm", "steps_ph",
        ]
        for col in new_cols:
            df[col] = None
        for j in range(INT_LENGTH):
            df[f"step_info_{str(j+1).zfill(2)}"] = None

        df_react_init = self.data.get("반응_init", pd.DataFrame())
        df_react_step = self.data.get("반응_step", pd.DataFrame())

        for i, row in df.iterrows():
            lot = row["lot_reacted"]

            # 수기운전일지 matching
            hand_list = self.data.get("수기운전일지", {}).get(line, [])
            for df_hand in hand_list:
                lot_hand = remove_trailing_non_digits(df_hand.iloc[2, 2])
                if lot_hand == lot:
                    df.at[i, "df_handrecorded"] = df_hand
                    break
            else:
                self.error_log.append(f"{line}:{i}: 수기운전일지 없음 ({lot})")
                _debug(f"수기운전일지 없음: {line}:{i} ({lot})", self.debug)

            # 반응_init matching by 생산LOT번호
            if not df_react_init.empty and "생산LOT번호" in df_react_init.columns:
                matched_init = df_react_init[df_react_init["생산LOT번호"] == lot]
                if len(matched_init) > 0:
                    df.at[i, "df_react_init"] = matched_init.iloc[0].to_dict()
                else:
                    self.error_log.append(f"{line}:{i}: 반응_init 없음 ({lot})")
                    _debug(f"반응_init 없음: {line}:{i} ({lot})", self.debug)

            # 반응_step matching by 생산LOT번호
            if not df_react_step.empty and "생산LOT번호" in df_react_step.columns:
                matched_steps = df_react_step[df_react_step["생산LOT번호"] == lot]
                if len(matched_steps) > 0:
                    self._parse_steps(df, i, matched_steps, line)
                else:
                    self.error_log.append(f"{line}:{i}: 반응_step 없음 ({lot})")
                    _debug(f"반응_step 없음: {line}:{i} ({lot})", self.debug)

        return df

    def _parse_steps(
        self, df: pd.DataFrame, i: int, steps_df: pd.DataFrame, line: str
    ) -> None:
        """Parse step data from 반응_step rows for a single LOT."""

        def to_float_list(col_name: str) -> list:
            return steps_df[col_name].astype(str).str.replace(",", "").astype(float).tolist()

        weights_metal = to_float_list("Unnamed: 5")
        weights_naoh = to_float_list("Unnamed: 8")
        weights_nh4oh = to_float_list("Unnamed: 11")

        # pH - case insensitive
        ph_col = None
        for c in steps_df.columns:
            if c.upper() == "PH":
                ph_col = c
                break
        list_ph = to_float_list(ph_col) if ph_col else []

        # RPM - no newline (교반기RPM)
        rpm_col = None
        for c in steps_df.columns:
            if "교반기" in c and "RPM" in c.upper():
                rpm_col = c
                break
        list_rpm = to_float_list(rpm_col) if rpm_col else []

        # Check for zeros
        if 0.0 in weights_metal + weights_naoh + weights_nh4oh + list_ph + list_rpm:
            self.error_log.append(f"  {i}: 투입량에 0 포함")
            _debug(f"투입량에 0 포함: row {i}", self.debug)
            return

        df.at[i, "weights_metal"] = weights_metal
        df.at[i, "weights_naoh"] = weights_naoh
        df.at[i, "weights_nh4oh"] = weights_nh4oh
        df.at[i, "steps_ph"] = list_ph
        df.at[i, "steps_rpm"] = list_rpm

        # Step numbers and times
        step_nums = steps_df["투입STEP"].astype(int).tolist()
        if step_nums[0] != 1:
            self.error_log.append(f"  {i}: 첫 STEP이 1이 아님")
            _debug(f"첫 STEP이 1이 아님: row {i}", self.debug)
            return

        df.at[i, "steps_num"] = step_nums
        df.at[i, "steps_time"] = steps_df["투입시간"].astype(float).tolist()

        # LOT numbers
        lots_metal = steps_df["Unnamed: 6"].dropna().unique().tolist()
        lots_naoh = steps_df["Unnamed: 9"].dropna().unique().tolist()
        lots_nh4oh = steps_df["Unnamed: 12"].dropna().unique().tolist()

        if not lots_metal or not lots_naoh or not lots_nh4oh:
            self.error_log.append(f"  {i}: LOT 번호 비어 있음")
            _debug(f"LOT 번호 비어 있음: row {i}", self.debug)
            return

        df.at[i, "lots_metal"] = lots_metal
        df.at[i, "lots_naoh"] = lots_naoh
        df.at[i, "lots_nh4oh"] = lots_nh4oh

        # step_info cumulative save
        dict_metal: dict = {"lot_metal": [], "weight": {}}
        dict_naoh: dict = {"lot_naoh": [], "weight": {}}
        dict_nh4oh: dict = {"lot_nh4oh": [], "weight": {}}

        for k, (_, step_row) in enumerate(steps_df.iterrows()):
            lot_m = step_row["Unnamed: 6"]
            lot_n = step_row["Unnamed: 9"]
            lot_nh = step_row["Unnamed: 12"]

            w_m = float(str(step_row["Unnamed: 5"]).replace(",", ""))
            w_n = float(str(step_row["Unnamed: 8"]).replace(",", ""))
            w_nh = float(str(step_row["Unnamed: 11"]).replace(",", ""))

            for lot_val, w_val, d in [
                (lot_m, w_m, dict_metal),
                (lot_n, w_n, dict_naoh),
                (lot_nh, w_nh, dict_nh4oh),
            ]:
                key = (
                    "lot_metal" if d is dict_metal
                    else ("lot_naoh" if d is dict_naoh else "lot_nh4oh")
                )
                if not isinstance(lot_val, str):
                    break
                if lot_val not in d[key]:
                    d[key].append(lot_val)
                    d["weight"][lot_val] = [w_val]
                else:
                    d["weight"][lot_val].append(w_val)

            step_save = {
                "metal": copy.deepcopy(dict_metal),
                "naoh": copy.deepcopy(dict_naoh),
                "nh4oh": copy.deepcopy(dict_nh4oh),
                "ph": list_ph[:k],
                "rpm": list_rpm[:k],
            }
            df.at[i, f"step_info_{str(k+1).zfill(2)}"] = step_save

    def _attach_melting(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
        """Attach melting data via 용해_metal DataFrame."""
        melt_cols = ["lots_coso4", "lots_mnso4", "lots_niso4",
                     "weight_coso4", "weight_mnso4", "weight_niso4"]
        for col in melt_cols:
            df[col] = None

        df_melt = self.data.get("용해_metal", pd.DataFrame())
        if df_melt.empty:
            return df

        for i, row in df.iterrows():
            lots_metal = row["lots_metal"]
            if lots_metal is None:
                continue

            coso4_lots, mnso4_lots, niso4_lots = [], [], []
            coso4_ws, mnso4_ws, niso4_ws = [], [], []
            all_found = True

            for lot_m in lots_metal:
                # Find rows in 용해_metal where 반응생산 LOT 번호 matches the reaction LOT
                # and 생산Lot번호 matches the metal lot
                matched = df_melt[df_melt["생산Lot번호"] == lot_m]
                if len(matched) == 0:
                    _debug(f"용해_metal 없음: lot_m={lot_m}", self.debug)
                    all_found = False
                    break

                for mat, lst_l, lst_w in [
                    ("황산코발트", coso4_lots, coso4_ws),
                    ("황산망간", mnso4_lots, mnso4_ws),
                    ("황산니켈", niso4_lots, niso4_ws),
                ]:
                    mat_rows = matched[matched["원료명"] == mat]
                    lst_l.append(mat_rows["LOT NO"].tolist())
                    lst_w.append(mat_rows["투입중량"].tolist())

            if all_found:
                df.at[i, "lots_coso4"] = coso4_lots
                df.at[i, "lots_mnso4"] = mnso4_lots
                df.at[i, "lots_niso4"] = niso4_lots
                df.at[i, "weight_coso4"] = coso4_ws
                df.at[i, "weight_mnso4"] = mnso4_ws
                df.at[i, "weight_niso4"] = niso4_ws

        return df

    def _attach_naoh(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
        """Attach NaOH material info."""
        df["df_naoh"] = None
        df_naoh_mat = self.data.get("원재료", {}).get("NAOH", pd.DataFrame())

        if df_naoh_mat.empty:
            return df

        for i, row in df.iterrows():
            lots = row["lots_naoh"]
            if lots is None:
                continue
            naoh_infos = []
            for lot in lots:
                prefix = lot.split("-")[0]
                matched = df_naoh_mat[
                    df_naoh_mat["입고LOT"].astype(str).str.startswith(prefix)
                ]
                if len(matched) > 0:
                    naoh_infos.append(matched.iloc[0].tolist())
                else:
                    _debug(f"NaOH 원재료 없음: lot={lot}, prefix={prefix}", self.debug)
            if len(naoh_infos) == len(lots):
                df.at[i, "df_naoh"] = naoh_infos

        return df

    def _attach_materials(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
        """Attach raw material inspection info for COSO4, MNSO4, NISO4."""
        mat_map = {
            "df_coso4": "COSO4",
            "df_mnso4": "MNSO4",
            "df_niso4": "NISO4",
        }

        for mat_col, mat_key in mat_map.items():
            df[mat_col] = None
            lots_col = mat_col.replace("df_", "lots_")
            df_mat = self.data.get("원재료", {}).get(mat_key, pd.DataFrame())

            if df_mat.empty:
                continue

            for i, row in df.iterrows():
                lot_groups = row.get(lots_col)
                if lot_groups is None:
                    continue

                all_infos = []
                for group in lot_groups:
                    group_infos = []
                    for lot in group:
                        prefix = lot.split("-")[0]
                        matched = df_mat[
                            df_mat["입고LOT"].astype(str).str.startswith(prefix)
                        ]
                        if len(matched) > 0:
                            group_infos.append(matched.iloc[0].tolist())
                        else:
                            _debug(
                                f"{mat_key} 원재료 없음: lot={lot}, prefix={prefix}",
                                self.debug,
                            )
                    if len(group_infos) == len(group):
                        all_infos.append(group_infos)

                if len(all_infos) == len(lot_groups):
                    df.at[i, mat_col] = all_infos

        return df
