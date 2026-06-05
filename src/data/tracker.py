import re
import copy
import math
import typing as tp
import pandas as pd
import numpy as np
from collections import defaultdict

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


def _parse_lot_parts(lot: str) -> tp.Tuple[str, str, str]:
    """lot_reacted에서 (라인번호, 반응기, 날짜YYMMDD) 추출."""
    parts = lot.split("-")
    m = re.match(r"(\d+)([A-Z]+)(\d{6})", parts[1])
    if not m:
        raise ValueError(f"{lot}: LOT 형식이 올바르지 않습니다.")
    return m.group(1), m.group(2), m.group(3)


def _yymmdd_to_lookup(date_str: str) -> str:
    """'2025-05-04 00:00:00' → '250504'"""
    date_part = date_str.split(" ")[0]  # 2025-05-04
    parts = date_part.split("-")
    parts[0] = parts[0][2:]  # 2025 → 25
    return "".join(parts)


def _facility_from_str(facility_str: str) -> str:
    """'반응기 A(GCOP01)' → 'A'"""
    return facility_str.split(" ")[1][0]


class TrackerRawData:
    """N86L/N86S 통합 LOT 트래커."""

    def __init__(
        self,
        data_excel: tp.DefaultDict,
        list_lines: tp.List[str],
        product_name: str,  # "N86L" or "N86S"
    ) -> None:
        self.data_excel = data_excel
        self.list_lines = list_lines
        self.list_lines_wanttoget = list_lines  # 레거시 호환
        self.product_name = product_name
        self.error_log: tp.List[str] = []
        self.dict_lines_tracked = self._track_all_lines()

    def _track_all_lines(self) -> tp.Dict[str, pd.DataFrame]:
        result = {}
        for line in self.list_lines:
            df_integrated = self.data_excel[line]["통합일지"][0]
            result[line] = self._track_line(df_integrated, line)
        return result

    def _track_line(self, df_integrated: pd.DataFrame, line: str) -> pd.DataFrame:
        df_lots = self._extract_lot_pairs(df_integrated, line)
        df = self._attach_reacted(df_lots, line)
        df = self._attach_naoh(df, line)
        df = self._attach_melting(df, line)
        df = self._attach_materials(df, line)
        return df

    def _extract_lot_pairs(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
        """통합일지에서 (lot_reacted, lot_target) 쌍 추출."""
        # 포맷 분기: N86S는 컬럼 구성이 두 가지
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

        # 2개 이상 제품 혼합 배치 제외
        lower = df_lots["lot_reacted"].str.lower()
        mixed = lower[lower.str.count(self.product_name.lower()) >= 2].index
        df_lots = df_lots.drop(mixed)

        df_lots = df_lots.dropna().reset_index(drop=True)
        df_lots = df_lots[df_lots["lot_target"].str.endswith("01")]
        df_lots["lot_reacted"] = df_lots["lot_reacted"].apply(extract_lot)
        return df_lots.reset_index(drop=True)

    def _attach_reacted(self, df_lots: pd.DataFrame, line: str) -> pd.DataFrame:
        """각 LOT에 반응투입스케쥴실적 + 수기운전일지 데이터 붙이기."""
        df = df_lots.copy()
        new_cols = [
            "num_line_from", "lot_reacted_found", "df_reacted", "df_handrecorded",
            "lots_metal", "lots_naoh", "lots_nh4oh",
            "weights_metal", "weights_naoh", "weights_nh4oh",
            "steps_num", "steps_time", "steps_rpm", "steps_ph",
        ]
        for col in new_cols:
            df[col] = None
        for j in range(INT_LENGTH):
            df[f"step_info_{str(j+1).zfill(2)}"] = None

        for i, row in df.iterrows():
            lot = row["lot_reacted"]

            # 수기운전일지 매칭
            for df_hand in self.data_excel[line].get("수기운전일지", []):
                lot_hand = remove_trailing_non_digits(df_hand.iloc[2, 2])
                if lot_hand == lot:
                    df.at[i, "df_handrecorded"] = df_hand
                    break
            else:
                self.error_log.append(f"{line}:{i}: 수기운전일지 없음 ({lot})")

            # 반응투입스케쥴실적 매칭
            try:
                target_line, target_facility, target_date = _parse_lot_parts(lot)
            except ValueError as e:
                self.error_log.append(str(e))
                continue

            df.at[i, "num_line_from"] = int(target_line)
            df_reacted_list = self.data_excel[f"{target_line}라인"].get("반응투입스케쥴실적", [])

            matched_reacted = None
            for d in df_reacted_list:
                if (_yymmdd_to_lookup(d["date"]) == target_date and
                        _facility_from_str(d["facility"]) == target_facility):
                    matched_reacted = d
                    break

            if matched_reacted is None:
                self.error_log.append(f"{line}:{i}: 반응투입스케쥴실적 없음 ({lot})")
                continue

            df.at[i, "df_reacted"] = matched_reacted["df"]

            sheet1 = matched_reacted["df"]["Sheet1"]
            idx = sheet1.loc[sheet1["작업시작일시"] == matched_reacted["date"]].index
            if len(idx):
                df.at[i, "lot_reacted_found"] = sheet1["생산 LOT 번호"][idx[0]]

            # Sheet2 파싱
            sheet2 = matched_reacted["df"]["Sheet2"].iloc[1:-1]
            self._parse_sheet2(df, i, sheet2, line, row)

        return df

    def _parse_sheet2(self, df: pd.DataFrame, i: int, sheet2: pd.DataFrame,
                      line: str, row: pd.Series) -> None:
        """Sheet2에서 투입량/LOT/스텝 정보 파싱."""
        def to_float_list(col):
            return sheet2[col].astype(str).str.replace(",", "").astype(float).tolist()

        weights_metal = to_float_list("Unnamed: 6")
        weights_naoh = to_float_list("Unnamed: 10")
        weights_nh4oh = to_float_list("Unnamed: 14")
        list_ph = to_float_list("PH")
        list_rpm = to_float_list("교반기\nRPM")

        # 0이 포함된 배치 제외
        if 0.0 in weights_metal + weights_naoh + weights_nh4oh + list_ph + list_rpm:
            self.error_log.append(f"  {i}: 투입량에 0 포함")
            return

        df.at[i, "weights_metal"] = weights_metal
        df.at[i, "weights_naoh"] = weights_naoh
        df.at[i, "weights_nh4oh"] = weights_nh4oh
        df.at[i, "steps_ph"] = list_ph
        df.at[i, "steps_rpm"] = list_rpm

        if sheet2["투입\nSTEP"].iloc[0] != 1:
            self.error_log.append(f"  {i}: 첫 STEP이 1이 아님")
            return

        df.at[i, "steps_num"] = sheet2["투입\nSTEP"].astype(int).tolist()
        df.at[i, "steps_time"] = sheet2["투입시간"].astype(float).tolist()

        # LOT 번호 — 1라인 NH4OH 컬럼 버그 수정 (is → ==)
        lots_metal = sheet2["Unnamed: 7"].dropna().unique().tolist()
        lots_naoh = sheet2["Unnamed: 11"].dropna().unique().tolist()
        nh4oh_col = "Unnamed: 19" if line == "1라인" else "Unnamed: 15"
        lots_nh4oh = sheet2[nh4oh_col].dropna().unique().tolist()

        if not lots_metal or not lots_naoh or not lots_nh4oh:
            self.error_log.append(f"  {i}: LOT 번호 비어 있음")
            return

        df.at[i, "lots_metal"] = lots_metal
        df.at[i, "lots_naoh"] = lots_naoh
        df.at[i, "lots_nh4oh"] = lots_nh4oh

        # step_info 누적 저장
        dict_metal: dict = {"lot_metal": [], "weight": {}}
        dict_naoh: dict = {"lot_naoh": [], "weight": {}}
        dict_nh4oh: dict = {"lot_nh4oh": [], "weight": {}}

        for k, (_, step_row) in enumerate(sheet2.iterrows()):
            if step_row.get("Unnamed: 4") is None or isinstance(step_row.get("Unnamed: 4"), float):
                break

            lot_m = step_row["Unnamed: 7"]
            lot_n = step_row["Unnamed: 11"]
            lot_nh = step_row[nh4oh_col]

            w_m = float(str(step_row["Unnamed: 6"]).replace(",", ""))
            w_n = float(str(step_row["Unnamed: 10"]).replace(",", ""))
            w_nh = float(str(step_row["Unnamed: 14"]).replace(",", ""))

            for lot_val, w_val, d in [(lot_m, w_m, dict_metal),
                                       (lot_n, w_n, dict_naoh),
                                       (lot_nh, w_nh, dict_nh4oh)]:
                key = "lot_metal" if d is dict_metal else ("lot_naoh" if d is dict_naoh else "lot_nh4oh")
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
                "ph": list_ph[:k],    # history of steps 0..k-1 (excludes current step — matches legacy)
                "rpm": list_rpm[:k],  # same: history before current step
            }
            df.at[i, f"step_info_{str(k+1).zfill(2)}"] = step_save

    def _attach_naoh(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
        """NaOH 원재료 정보 붙이기."""
        df["df_naoh"] = None
        for i, row in df.iterrows():
            lots = row["lots_naoh"]
            if lots is None:
                continue
            naoh_infos = []
            for lot in lots:
                lot = lot.split("-")[0]
                for df_mat in self.data_excel["원재료"].get("가성소다", []):
                    if lot in df_mat["입고LOT"].values:
                        info = df_mat[df_mat["입고LOT"] == lot].iloc[0].tolist()
                        naoh_infos.append(info)
                        break
            if len(naoh_infos) == len(lots):
                df.at[i, "df_naoh"] = naoh_infos
        return df

    def _attach_melting(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
        """용해작업실적 붙이기."""
        melt_cols = ["df_melted", "lots_coso4", "lots_mnso4", "lots_niso4",
                     "weight_coso4", "weight_mnso4", "weight_niso4"]
        for col in melt_cols:
            df[col] = None

        for i, row in df.iterrows():
            lots_metal = row["lots_metal"]
            line_num = row["num_line_from"]
            if lots_metal is None or line_num is None:
                continue

            melted_list, coso4_lots, mnso4_lots, niso4_lots = [], [], [], []
            coso4_ws, mnso4_ws, niso4_ws = [], [], []

            for lot_m in lots_metal:
                for d in self.data_excel[f"{int(line_num)}라인"].get("용해작업실적", []):
                    if lot_m == d["lot_melted"]:
                        melted_list.append(d["df"])
                        sheet2 = d["df"]["Sheet2"]
                        for mat, lst_l, lst_w in [
                            ("황산코발트", coso4_lots, coso4_ws),
                            ("황산망간", mnso4_lots, mnso4_ws),
                            ("황산니켈", niso4_lots, niso4_ws),
                        ]:
                            idx = sheet2[sheet2["원료명"] == mat].index
                            lst_l.append(sheet2["LOT NO"][idx].tolist())
                            lst_w.append(sheet2["투입중량"][idx].tolist())
                        break

            if len(melted_list) == len(lots_metal):
                df.at[i, "df_melted"] = melted_list
                df.at[i, "lots_coso4"] = coso4_lots
                df.at[i, "lots_mnso4"] = mnso4_lots
                df.at[i, "lots_niso4"] = niso4_lots
                df.at[i, "weight_coso4"] = coso4_ws
                df.at[i, "weight_mnso4"] = mnso4_ws
                df.at[i, "weight_niso4"] = niso4_ws

        return df

    def _attach_materials(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
        """원재료 검사 정보 붙이기."""
        for mat_col, mat_key in [("df_coso4", "황산코발트"), ("df_mnso4", "황산망간"),
                                   ("df_niso4", "황산니켈")]:
            df[mat_col] = None
            lots_col = mat_col.replace("df_", "lots_")

            for i, row in df.iterrows():
                lot_groups = row.get(lots_col)
                if lot_groups is None:
                    continue

                all_infos = []
                for group in lot_groups:
                    group_infos = []
                    for lot in group:
                        lot = lot.split("-")[0]
                        for df_mat in self.data_excel["원재료"].get(mat_key, []):
                            if lot in df_mat["입고LOT"].values:
                                info = df_mat[df_mat["입고LOT"] == lot].iloc[0].tolist()
                                group_infos.append(info)
                                break
                    if len(group_infos) == len(group):
                        all_infos.append(group_infos)

                if len(all_infos) == len(lot_groups):
                    df.at[i, mat_col] = all_infos

        return df
