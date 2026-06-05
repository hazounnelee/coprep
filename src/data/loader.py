import os
import typing as tp
from collections import defaultdict
import pandas as pd

TASKS_1SHEET = ["통합일지", "수기운전일지", "제품시험검사이력", "제품검사판정이력"]
TASKS_2SHEET = ["용해작업실적"]
TASKS_3SHEET = ["반응투입스케쥴실적"]
TASKS_MATERIAL = ["가성소다", "황산니켈", "황산망간", "황산코발트", "암모니아", "검사이력"]
TASKS_TESTRESULT = ["검사이력", "제품검사판정이력", "제품시험검사이력"]


def _get_xlsx_paths(folder: str) -> tp.List[tp.Tuple[str, tp.List[str]]]:
    result = []
    for path, _, filenames in os.walk(folder):
        xlsx = [f for f in filenames if f.endswith(".xlsx")]
        if xlsx:
            result.append((path, xlsx))
    return result


def _load_1sheet(folder: str, filenames: tp.List[str]) -> tp.List[pd.DataFrame]:
    return [pd.read_excel(os.path.join(folder, f), engine="openpyxl") for f in filenames]


def _load_2sheet(folder: str, filenames: tp.List[str]) -> tp.List[dict]:
    results = []
    for f in filenames:
        sheets = pd.read_excel(os.path.join(folder, f), engine="openpyxl", sheet_name=None)
        sheet1 = sheets["Sheet1"]
        idx = sheet1.loc[sheet1["No"] == "▶"].index
        if len(idx) != 1:
            raise ValueError(f"{f}: Sheet1에 선택 마커(▶)가 1개여야 합니다.")
        lot_melted = sheet1["생산Lot번호"][idx[0]]
        results.append({"df": sheets, "lot_melted": lot_melted})
    return results


def _load_3sheet(folder: str, filenames: tp.List[str]) -> tp.List[dict]:
    results = []
    for f in filenames:
        sheets = pd.read_excel(os.path.join(folder, f), engine="openpyxl", sheet_name=None)
        sheet1 = sheets["Sheet1"]
        idx = sheet1.loc[sheet1["No"] == "▶"].index
        if len(idx) != 1:
            raise ValueError(f"{f}: Sheet1에 선택 마커(▶)가 1개여야 합니다.")
        date = str(sheet1["작업시작일시"][idx[0]])
        facility = str(sheet1["설비명"][idx[0]])
        results.append({"df": sheets, "date": date, "facility": facility})
    return results


def get_alldata(path_folder: str) -> tp.DefaultDict:
    all_paths = _get_xlsx_paths(path_folder)
    data: tp.DefaultDict = defaultdict(dict)

    for folder, filenames in all_paths:
        task_name = folder.split("/")[-1]
        category = folder.split("/")[-2]

        if category.endswith("라인"):  # "1라인"..."10라인" — substring match covers all lines
            if task_name in TASKS_1SHEET:
                data[category][task_name] = _load_1sheet(folder, filenames)
            elif task_name in TASKS_2SHEET:
                data[category][task_name] = _load_2sheet(folder, filenames)
            elif task_name in TASKS_3SHEET:
                data[category][task_name] = _load_3sheet(folder, filenames)

        elif category == "원재료":
            if task_name in TASKS_MATERIAL:
                data[category][task_name] = _load_1sheet(folder, filenames)

        elif category == "검사이력":
            if task_name in TASKS_TESTRESULT:
                data[category][task_name] = _load_1sheet(folder, filenames)

    return data
