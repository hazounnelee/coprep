# Data Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the preprocessing pipeline to the new consolidated data structure, implement material weighted-sum features, and add debug diagnostics.

**Architecture:** Minimal adaptation of existing loader → tracker → preprocessor pipeline. Each module's internal logic is updated for new file formats while preserving the overall flow. Loader returns a flat dict instead of nested defaultdict. Tracker outputs a single DataFrame instead of per-line dicts.

**Tech Stack:** Python, pandas, numpy, openpyxl, xgboost, pytest

---

### Task 1: Rename STEP_METERIAL_ → STEP_MATERIAL_ in postprocessor

**Files:**
- Modify: `src/model/postprocessor.py`
- Modify: `tests/test_postprocessor.py`

- [ ] **Step 1: Update constants and patterns in postprocessor.py**

Replace all `METERIAL` references with `MATERIAL`:

```python
# src/model/postprocessor.py line 7-8
LIST_PREFIX = ["STEP_WEIGHT_", "STEP_RPM_", "STEP_PH_", "STEP_SIZE_", "STEP_MATERIAL_"]
PATTERN_MATERIAL = re.compile(r"^STEP_MATERIAL_(\d+)")

# line 31-32: update reference in _build_step_col_map
        if col.startswith("STEP_MATERIAL_"):
            m = PATTERN_MATERIAL.match(col)

# line 85: update filter_material
    mat_cols = df_x.filter(like="STEP_MATERIAL").columns.tolist()

# line 101: update postprocess_by_product
        for prefix in ["STEP_RPM_", "STEP_PH_", "STEP_MATERIAL_"]:
```

- [ ] **Step 2: Run existing tests**

Run: `pytest tests/test_postprocessor.py -v`
Expected: PASS (existing tests don't use STEP_METERIAL_ columns)

- [ ] **Step 3: Commit**

```bash
git add src/model/postprocessor.py
git commit -m "refactor: rename STEP_METERIAL_ to STEP_MATERIAL_"
```

---

### Task 2: Rewrite loader.py for new data structure

**Files:**
- Modify: `src/data/loader.py`
- Modify: `tests/test_loader.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write tests for new loader functions**

Replace `tests/test_loader.py` entirely:

```python
import os
import pytest
import pandas as pd
import numpy as np
from src.data.loader import (
    get_alldata, _load_integrated, _load_handrecorded,
    _load_melt_metal, _load_melt_ord,
    _load_react_init, _load_react_step,
    _load_materials, MATERIAL_PROPS,
)


@pytest.fixture
def data_folder(tmp_path):
    """Create a minimal new-structure data folder with dummy Excel files."""
    # 통합일지
    integ_dir = tmp_path / "통합일지"
    integ_dir.mkdir()
    df_integ = pd.DataFrame({
        "근무자": [None]*4 + ["홍길동", "홍길동"],
        "No.": [None]*4 + ["N86L-1A250501-02", "N86L-1A250502-02"],
        "포장 Lot No": [None]*4 + ["PKG-0001-01", "PKG-0002-01"],
        "생산품 구분\n(양산/DOE)": [None]*4 + ["양산", "양산"],
        "포장 일자": [None]*6,
        "포장 시간": [None]*6,
    })
    df_integ.to_excel(integ_dir / "복사본 2024년도 1라인-전구체2단계 통합일지.xlsx",
                      index=False, engine="openpyxl")

    # 수기운전일지
    hand_dir = tmp_path / "수기운전일지" / "1라인"
    hand_dir.mkdir(parents=True)
    rows = [[None]*7 for _ in range(12)]
    rows[2][2] = "N86L-1A250501-02"
    for t in [120, 300, 600]:
        rows.append([str(t), None, "6.0", "7.0", "8.0", "9.0", "10.0"])
    df_hand = pd.DataFrame(rows)
    df_hand.to_excel(hand_dir / "운전일지_001.xlsx", index=False, engine="openpyxl")

    # 용해작업실적
    melt_dir = tmp_path / "용해작업실적"
    melt_dir.mkdir()
    df_melt_metal = pd.DataFrame({
        "1.화면항목": ["생산Lot번호", "METAL-LOT-001", "METAL-LOT-001"],
        "Unnamed: 1": [None, None, None],
        "Unnamed: 2": [None, None, None],
        "Unnamed: 3": ["설비작업번호", None, None],
        "Unnamed: 4": [None, None, None],
        "Unnamed: 5": ["투자차수", None, None],
        "Unnamed: 6": ["원료코드", None, None],
        "Unnamed: 7": ["원료명", "황산니켈", "황산코발트"],
        "Unnamed: 8": ["LOT NO", "NISO4-001", "COSO4-001"],
        "Unnamed: 9": ["포장중량", None, None],
        "Unnamed: 10": ["측정중량", None, None],
        "Unnamed: 11": ["소분 후 중량", None, None],
        "Unnamed: 12": ["포장재준량", None, None],
        "Unnamed: 13": ["투입중량", "500.0", "100.0"],
        "Unnamed: 24": ["반응생산 LOT 번호", "N86L-1A250501-02", "N86L-1A250501-02"],
    })
    df_melt_metal.to_excel(melt_dir / "MELT_WRK_ORD_METAL.xlsx",
                           index=False, engine="openpyxl")

    df_melt_ord = pd.DataFrame({
        "1.화면항목": ["", ""],
        "Unnamed: 4": ["생산Lot번호", "METAL-LOT-001"],
    })
    df_melt_ord.to_excel(melt_dir / "MELT_WRK_ORD.xlsx",
                         index=False, engine="openpyxl")

    # 반응투입스케줄
    react_dir = tmp_path / "반응투입스케줄"
    react_dir.mkdir()
    df_init = pd.DataFrame({
        "생산LOT번호": ["N86L-1A250501-02"],
        "초기시작일시": ["2025-05-01 00:00:00"],
        "초기종료일시": ["2025-05-01 08:00:00"],
        "작업시간": [480],
        "순수온도": [50.0],
        "NAOH순수투입중량(kg)": [200.0],
        "NH4OH순수투입중량(kg)": [100.0],
        "순수투입중량(kg)": [1200.0],
        "용존산소량": [0.5],
        "Ph": [11.0],
        "N2주입유량": [5.0],
        "N2 PURGE시간": [30.0],
    })
    df_init.to_excel(react_dir / "반응투입_LOT_INIT.xlsx",
                     index=False, engine="openpyxl")

    step_header = pd.DataFrame([
        ["N86L-1A250501-02", 1, 60, 60, None, "150.0", "METAL-LOT-001",
         None, "80.0", "NAOH-LOT-001", None, "50.0", "NH4OH-LOT-001",
         "11.5", "700"],
        ["N86L-1A250501-02", 2, 60, 120, None, "155.0", "METAL-LOT-001",
         None, "82.0", "NAOH-LOT-001", None, "51.0", "NH4OH-LOT-001",
         "11.6", "710"],
    ], columns=[
        "생산LOT번호", "투입STEP", "투입시간", "Unnamed: 3",
        "Metal Solution", "Unnamed: 5", "Unnamed: 6",
        "NAOH", "Unnamed: 8", "Unnamed: 9",
        "NH4OH", "Unnamed: 11", "Unnamed: 12",
        "PH", "교반기RPM",
    ])
    step_header.to_excel(react_dir / "반응투입_LOT_STEP.xlsx",
                         index=False, engine="openpyxl")

    # 원재료
    mat_dir = tmp_path / "원재료"
    mat_dir.mkdir()
    for mat_type, lot in [("COSO4", "COSO4-001"), ("NISO4", "NISO4-001"),
                          ("MNSO4", "MNSO4-001"), ("NAOH", "NAOH-LOT"),
                          ("NH4OH", "NH4OH-LOT")]:
        header1 = ["No", "입고일", "입고LOT", "공급업체", "제조업체",
                    "제조원LOT", "입고량", "단위", "판정", "검사분류",
                    "성분", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13",
                    "Unnamed: 14", "Unnamed: 15", "Unnamed: 16", "Unnamed: 17",
                    "Unnamed: 18", "Unnamed: 19", "Unnamed: 20", "Unnamed: 21",
                    "Unnamed: 22", "Unnamed: 23", "Unnamed: 24", "Unnamed: 25"]
        row_type = [None]*10 + ["검사유형", "Chemical Composition (C01)",
                    None, None, None, None, None, None, None, None, None, None,
                    "Initial PH (C03)", None, "Magnetic Impurities (C04)", None]
        row_item = [None]*10 + ["검사항목", "Co", "Al", "Ca", "Cd", "Cu",
                    "Fe", "Li", "Mg", "Mn", "Na", "Ni", "pH",
                    "Total(자성이물)", "Cr", "Fe"]
        row_data = [1, "2025-01-01", lot, "업체A", "제조A", "MFG-001",
                    1000, "kg", "합격", "수입검사",
                    None, 99.5, 0.01, 0.02, 0.001, 0.005,
                    0.01, 0.001, 0.005, 30.0, 0.1, 55.0, 11.5,
                    0.001, 0.0001, 0.0001]
        df_mat = pd.DataFrame([header1, row_type, row_item, row_data])
        fname = f"MATR_{mat_type}_00001_원료검사판정이력_220401_260408.xlsx"
        df_mat.to_excel(mat_dir / fname, index=False, header=False, engine="openpyxl")

    return str(tmp_path)


def test_load_integrated(data_folder):
    result = _load_integrated(data_folder)
    assert "1라인" in result
    df = result["1라인"]
    assert "No." in df.columns or "No" in df.columns


def test_load_handrecorded(data_folder):
    result = _load_handrecorded(data_folder)
    assert "1라인" in result
    assert len(result["1라인"]) >= 1


def test_load_melt_metal(data_folder):
    df = _load_melt_metal(data_folder)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_react_init(data_folder):
    df = _load_react_init(data_folder)
    assert "생산LOT번호" in df.columns
    assert "작업시간" in df.columns


def test_load_react_step(data_folder):
    df = _load_react_step(data_folder)
    assert len(df) > 0


def test_load_materials(data_folder):
    result = _load_materials(data_folder)
    assert "COSO4" in result
    assert "NAOH" in result
    df = result["COSO4"]
    assert "입고LOT" in df.columns


def test_get_alldata_returns_all_keys(data_folder):
    data = get_alldata(data_folder)
    for key in ["통합일지", "수기운전일지", "용해_metal", "용해_ord",
                "반응_init", "반응_step", "원재료"]:
        assert key in data


def test_debug_mode_no_crash(data_folder):
    data = get_alldata(data_folder, debug=True)
    assert data is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL (new functions don't exist yet)

- [ ] **Step 3: Implement new loader.py**

Replace `src/data/loader.py` entirely:

```python
import os
import re
import typing as tp
import pandas as pd


MATERIAL_TYPE_MAP = {
    "NAOH": "NAOH",
    "COSO4": "COSO4",
    "MNSO4": "MNSO4",
    "NISO4": "NISO4",
    "NH4OH": "NH4OH",
}

MATERIAL_PROPS = {
    "COSO4": {"Chemical Composition (C01)": ["Co"]},
    "MNSO4": {"Chemical Composition (C01)": ["Mn"], "Initial PH (C03)": ["pH"]},
    "NISO4": {"Chemical Composition (C01)": ["Ni"]},
    "NAOH": {"Chemical Composition (C01)": ["NaOH"]},
    "NH4OH": {},
}


def _debug_log(debug: bool, msg: str) -> None:
    if debug:
        print(f"[DEBUG] loader: {msg}")


def _find_file(folder: str, pattern: str, debug: bool = False) -> tp.Optional[str]:
    if not os.path.isdir(folder):
        _debug_log(debug, f"폴더 없음: {folder}")
        return None
    for f in os.listdir(folder):
        if re.search(pattern, f, re.IGNORECASE) and f.endswith(".xlsx"):
            return os.path.join(folder, f)
    _debug_log(debug, f"패턴 '{pattern}'에 맞는 파일 없음: {folder}")
    return None


def _find_files(folder: str, pattern: str, debug: bool = False) -> tp.List[str]:
    if not os.path.isdir(folder):
        _debug_log(debug, f"폴더 없음: {folder}")
        return []
    return [os.path.join(folder, f)
            for f in os.listdir(folder)
            if re.search(pattern, f, re.IGNORECASE) and f.endswith(".xlsx")]


def _load_integrated(path_folder: str, debug: bool = False) -> tp.Dict[str, pd.DataFrame]:
    folder = os.path.join(path_folder, "통합일지")
    result = {}
    if not os.path.isdir(folder):
        _debug_log(debug, "통합일지 폴더 없음")
        return result
    for fname in os.listdir(folder):
        if not fname.endswith(".xlsx"):
            continue
        m = re.search(r"(\d+)라인", fname)
        if not m:
            _debug_log(debug, f"통합일지 파일에서 라인 추출 실패: {fname}")
            continue
        line = f"{m.group(1)}라인"
        path = os.path.join(folder, fname)
        df = pd.read_excel(path, engine="openpyxl")
        _debug_log(debug, f"통합일지 {line}: {len(df)} rows, columns={list(df.columns)}")
        result[line] = df
    return result


def _load_handrecorded(path_folder: str, debug: bool = False) -> tp.Dict[str, tp.List[pd.DataFrame]]:
    folder = os.path.join(path_folder, "수기운전일지")
    result = {}
    if not os.path.isdir(folder):
        _debug_log(debug, "수기운전일지 폴더 없음")
        return result
    for sub in sorted(os.listdir(folder)):
        sub_path = os.path.join(folder, sub)
        if not os.path.isdir(sub_path) or "라인" not in sub:
            continue
        dfs = []
        for fname in os.listdir(sub_path):
            if fname.endswith(".xlsx"):
                df = pd.read_excel(os.path.join(sub_path, fname), engine="openpyxl")
                dfs.append(df)
        result[sub] = dfs
        _debug_log(debug, f"수기운전일지 {sub}: {len(dfs)} files")
    return result


def _load_melt_metal(path_folder: str, debug: bool = False) -> pd.DataFrame:
    folder = os.path.join(path_folder, "용해작업실적")
    path = _find_file(folder, r"MELT_WRK_ORD_METAL", debug)
    if path is None:
        return pd.DataFrame()
    df = pd.read_excel(path, header=1, engine="openpyxl")
    _debug_log(debug, f"용해_metal: {len(df)} rows, columns={list(df.columns)}")
    return df


def _load_melt_ord(path_folder: str, debug: bool = False) -> pd.DataFrame:
    folder = os.path.join(path_folder, "용해작업실적")
    path = _find_file(folder, r"MELT_WRK_ORD\.xlsx$", debug)
    if path is None:
        return pd.DataFrame()
    df = pd.read_excel(path, header=1, engine="openpyxl")
    _debug_log(debug, f"용해_ord: {len(df)} rows, columns={list(df.columns)}")
    return df


def _load_react_init(path_folder: str, debug: bool = False) -> pd.DataFrame:
    folder = os.path.join(path_folder, "반응투입스케줄")
    path = _find_file(folder, r"반응투입_LOT_INIT", debug)
    if path is None:
        return pd.DataFrame()
    df = pd.read_excel(path, engine="openpyxl")
    _debug_log(debug, f"반응_init: {len(df)} rows, columns={list(df.columns)}")
    return df


def _load_react_step(path_folder: str, debug: bool = False) -> pd.DataFrame:
    folder = os.path.join(path_folder, "반응투입스케줄")
    path = _find_file(folder, r"반응투입_LOT_STEP", debug)
    if path is None:
        return pd.DataFrame()
    df = pd.read_excel(path, engine="openpyxl")
    _debug_log(debug, f"반응_step: {len(df)} rows, columns={list(df.columns)}")
    return df


def _parse_material_file(path: str, mat_type: str, debug: bool = False) -> pd.DataFrame:
    df_raw = pd.read_excel(path, header=None, engine="openpyxl")
    if len(df_raw) < 4:
        _debug_log(debug, f"원재료 {mat_type}: 행이 4개 미만")
        return pd.DataFrame()

    row_header = df_raw.iloc[0]
    row_type = df_raw.iloc[1]
    row_item = df_raw.iloc[2]
    df_data = df_raw.iloc[3:].reset_index(drop=True)

    lot_col_idx = None
    for idx, val in enumerate(row_header):
        if isinstance(val, str) and val.strip().lower() == "입고lot":
            lot_col_idx = idx
            break

    if lot_col_idx is None:
        _debug_log(debug, f"원재료 {mat_type}: '입고LOT' 컬럼 없음. headers={list(row_header)}")
        return pd.DataFrame()

    result = pd.DataFrame({"입고LOT": df_data.iloc[:, lot_col_idx]})

    props = MATERIAL_PROPS.get(mat_type, {})
    for type_name, item_names in props.items():
        for item_name in item_names:
            col_idx = _find_material_col(row_type, row_item, type_name, item_name)
            if col_idx is not None:
                col_label = f"{type_name}_{item_name}"
                result[col_label] = pd.to_numeric(df_data.iloc[:, col_idx], errors="coerce")
            else:
                _debug_log(debug,
                    f"원재료 {mat_type}: '{type_name} > {item_name}' 컬럼 없음")

    return result


def _find_material_col(row_type: pd.Series, row_item: pd.Series,
                       type_name: str, item_name: str) -> tp.Optional[int]:
    current_type = None
    for idx in range(len(row_type)):
        t = row_type.iloc[idx]
        if isinstance(t, str) and t.strip():
            current_type = t.strip()
        if (current_type and current_type.lower() == type_name.lower()
                and isinstance(row_item.iloc[idx], str)
                and row_item.iloc[idx].strip().lower() == item_name.lower()):
            return idx
    return None


def _load_materials(path_folder: str, debug: bool = False) -> tp.Dict[str, pd.DataFrame]:
    folder = os.path.join(path_folder, "원재료")
    result = {}
    if not os.path.isdir(folder):
        _debug_log(debug, "원재료 폴더 없음")
        return result

    for fname in os.listdir(folder):
        if not fname.endswith(".xlsx"):
            continue
        m = re.match(r"MATR_([A-Za-z0-9]+)_", fname)
        if not m:
            continue
        mat_type = m.group(1).upper()
        if mat_type not in MATERIAL_TYPE_MAP:
            _debug_log(debug, f"원재료 알 수 없는 종류: {mat_type} ({fname})")
            continue

        path = os.path.join(folder, fname)
        df = _parse_material_file(path, mat_type, debug)
        if mat_type in result and not df.empty:
            result[mat_type] = pd.concat([result[mat_type], df], ignore_index=True)
        elif not df.empty:
            result[mat_type] = df

    return result


def get_alldata(path_folder: str, debug: bool = False) -> dict:
    return {
        "통합일지": _load_integrated(path_folder, debug),
        "수기운전일지": _load_handrecorded(path_folder, debug),
        "용해_metal": _load_melt_metal(path_folder, debug),
        "용해_ord": _load_melt_ord(path_folder, debug),
        "반응_init": _load_react_init(path_folder, debug),
        "반응_step": _load_react_step(path_folder, debug),
        "원재료": _load_materials(path_folder, debug),
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: rewrite loader for new consolidated data structure"
```

---

### Task 3: Rewrite tracker.py

**Files:**
- Modify: `src/data/tracker.py`
- Modify: `tests/test_tracker.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Update conftest.py with new fixture format**

Replace `tests/conftest.py` entirely. The new fixture must produce data in the format returned by the new `get_alldata`:

```python
import pytest
import pandas as pd
import numpy as np


def make_dummy_통합일지(n_lots: int = 3, product: str = "N86L") -> pd.DataFrame:
    rows = []
    for _ in range(4):
        rows.append({"No.": None, "포장 Lot No": None,
                     "생산품 구분\n(양산/DOE)": None})
    for i in range(n_lots):
        lot_reacted = f"{product}-1A25050{i+1}-02"
        lot_target = f"PKG-{i+1:04d}-01"
        rows.append({"No.": lot_reacted, "포장 Lot No": lot_target,
                     "생산품 구분\n(양산/DOE)": "양산"})
    return pd.DataFrame(rows)


def make_dummy_수기운전일지(lot: str, n_measurements: int = 10) -> pd.DataFrame:
    np.random.seed(hash(lot) % 2**31)
    rows = []
    for r in range(12):
        if r == 2:
            row = [None, None, lot, None, None, None, None]
        else:
            row = [None] * 7
        rows.append(row)
    times = sorted(np.random.choice(range(60, 3200, 60), n_measurements, replace=False))
    for t in times:
        d50 = 8.0 + np.random.randn() * 0.3
        row = [str(t), None, str(round(d50 * 0.8, 2)), str(round(d50 * 0.9, 2)),
               str(round(d50, 2)), str(round(d50 * 1.1, 2)), str(round(d50 * 1.2, 2))]
        rows.append(row)
    return pd.DataFrame(rows)


def make_dummy_반응_init(lots: list) -> pd.DataFrame:
    rows = []
    for lot in lots:
        rows.append({
            "생산LOT번호": lot,
            "초기시작일시": "2025-05-01 00:00:00",
            "초기종료일시": "2025-05-01 08:00:00",
            "작업시간": 480,
            "순수온도": 50.0,
            "NAOH순수투입중량(kg)": 200.0,
            "NH4OH순수투입중량(kg)": 100.0,
            "순수투입중량(kg)": 1200.0,
            "용존산소량": 0.5,
            "Ph": 11.0,
            "N2주입유량": 5.0,
            "N2 PURGE시간": 30.0,
        })
    return pd.DataFrame(rows)


def make_dummy_반응_step(lots: list, n_steps: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for lot in lots:
        for step in range(1, n_steps + 1):
            rows.append({
                "생산LOT번호": lot,
                "투입STEP": step,
                "투입시간": 60.0,
                "Unnamed: 3": step * 60,
                "Metal Solution": None,
                "Unnamed: 5": f"{150.0 + np.random.randn() * 5:.1f}",
                "Unnamed: 6": "METAL-LOT-001",
                "NAOH": None,
                "Unnamed: 8": f"{80.0 + np.random.randn() * 2:.1f}",
                "Unnamed: 9": "NAOH-LOT-001",
                "NH4OH": None,
                "Unnamed: 11": f"{50.0 + np.random.randn() * 1:.1f}",
                "Unnamed: 12": "NH4OH-LOT-001",
                "PH": f"{11.5 + np.random.randn() * 0.1:.2f}",
                "교반기RPM": f"{700 + np.random.randint(-10, 10)}",
            })
    return pd.DataFrame(rows)


def make_dummy_용해_metal(lots_metal: list, reaction_lots: list) -> pd.DataFrame:
    rows = []
    for metal_lot, react_lot in zip(lots_metal, reaction_lots):
        for mat, mat_lot, weight in [
            ("황산니켈", "NISO4-001", "500.0"),
            ("황산코발트", "COSO4-001", "100.0"),
            ("황산망간", "MNSO4-001", "200.0"),
        ]:
            rows.append({
                "생산Lot번호": metal_lot,
                "원료명": mat,
                "LOT NO": mat_lot,
                "투입중량": weight,
                "반응생산 LOT 번호": react_lot,
            })
    return pd.DataFrame(rows)


def make_dummy_원재료() -> dict:
    result = {}
    for mat_type, lot_prefix in [("COSO4", "COSO4-001"), ("MNSO4", "MNSO4-001"),
                                  ("NISO4", "NISO4-001"), ("NAOH", "NAOH-LOT"),
                                  ("NH4OH", "NH4OH-LOT")]:
        data = {"입고LOT": [lot_prefix]}
        if mat_type == "COSO4":
            data["Chemical Composition (C01)_Co"] = [99.5]
        elif mat_type == "MNSO4":
            data["Chemical Composition (C01)_Mn"] = [30.0]
            data["Initial PH (C03)_pH"] = [3.5]
        elif mat_type == "NISO4":
            data["Chemical Composition (C01)_Ni"] = [55.0]
        elif mat_type == "NAOH":
            data["Chemical Composition (C01)_NaOH"] = [48.0]
        result[mat_type] = pd.DataFrame(data)
    return result


@pytest.fixture
def dummy_data():
    """New-format data dict matching get_alldata output."""
    lots = ["N86L-1A250501-02", "N86L-1A250502-02", "N86L-1A250503-02"]
    return {
        "통합일지": {"1라인": make_dummy_통합일지(n_lots=3)},
        "수기운전일지": {"1라인": [make_dummy_수기운전일지(lot) for lot in lots]},
        "반응_init": make_dummy_반응_init(lots),
        "반응_step": make_dummy_반응_step(lots),
        "용해_metal": make_dummy_용해_metal(
            ["METAL-LOT-001"] * 3, lots),
        "용해_ord": pd.DataFrame({"생산Lot번호": ["METAL-LOT-001"]}),
        "원재료": make_dummy_원재료(),
    }


@pytest.fixture
def seq_1_6_path(tmp_path):
    p = tmp_path / "seq_1_6.txt"
    lines = [(i + 1, 60 if i not in [4, 5, 6, 7, 10, 11, 21, 22] else 30)
             for i in range(53)]
    p.write_text("\n".join(f"{s}\t{t}" for s, t in lines))
    return str(p)
```

- [ ] **Step 2: Write new tracker tests**

Replace `tests/test_tracker.py`:

```python
import pytest
from src.data.tracker import TrackerRawData, extract_lot, remove_trailing_non_digits


def test_extract_lot_parses_valid():
    assert extract_lot("N86L-1A250504-02") == "N86L-1A250504-02"


def test_remove_trailing_non_digits():
    assert remove_trailing_non_digits("N86L-1A250504-02A") == "N86L-1A250504-02"
    assert remove_trailing_non_digits("N86L-1A250504-02") == "N86L-1A250504-02"


def test_tracker_returns_single_dataframe(dummy_data):
    tracker = TrackerRawData(
        data=dummy_data,
        list_lines=["1라인"],
        product_name="N86L",
    )
    df = tracker.df_tracked
    assert "lot_target" in df.columns
    assert "lot_reacted" in df.columns
    assert len(df) > 0


def test_tracker_has_step_info(dummy_data):
    tracker = TrackerRawData(
        data=dummy_data,
        list_lines=["1라인"],
        product_name="N86L",
    )
    df = tracker.df_tracked
    step_cols = [c for c in df.columns if c.startswith("step_info_")]
    assert len(step_cols) > 0


def test_tracker_has_weights(dummy_data):
    tracker = TrackerRawData(
        data=dummy_data,
        list_lines=["1라인"],
        product_name="N86L",
    )
    df = tracker.df_tracked
    assert "weights_metal" in df.columns
    assert "weights_naoh" in df.columns
    assert "weights_nh4oh" in df.columns


def test_tracker_has_material_properties(dummy_data):
    tracker = TrackerRawData(
        data=dummy_data,
        list_lines=["1라인"],
        product_name="N86L",
    )
    df = tracker.df_tracked
    assert "df_naoh" in df.columns or "mat_naoh" in df.columns


def test_tracker_debug_mode(dummy_data):
    tracker = TrackerRawData(
        data=dummy_data,
        list_lines=["1라인"],
        product_name="N86L",
        debug=True,
    )
    assert tracker.df_tracked is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_tracker.py -v`
Expected: FAIL

- [ ] **Step 4: Implement new tracker.py**

Replace `src/data/tracker.py` entirely:

```python
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


def _debug_log(debug: bool, msg: str) -> None:
    if debug:
        print(f"[DEBUG] tracker: {msg}")


class TrackerRawData:

    def __init__(
        self,
        data: tp.Dict,
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
        df = self._extract_all_lot_pairs()
        df = self._attach_reacted(df)
        df = self._attach_handrecorded(df)
        df = self._attach_melting(df)
        df = self._attach_materials(df)
        return df

    def _extract_all_lot_pairs(self) -> pd.DataFrame:
        all_lots = []
        for line in self.list_lines:
            df_integ = self.data["통합일지"].get(line)
            if df_integ is None:
                _debug_log(self.debug, f"{line}: 통합일지 없음")
                continue
            df_pairs = self._extract_lot_pairs(df_integ, line)
            all_lots.append(df_pairs)
        if not all_lots:
            return pd.DataFrame(columns=["lot_reacted", "lot_target"])
        return pd.concat(all_lots, ignore_index=True)

    def _extract_lot_pairs(self, df: pd.DataFrame, line: str) -> pd.DataFrame:
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

        lower = df_lots["lot_reacted"].str.lower()
        mixed = lower[lower.str.count(self.product_name.lower()) >= 2].index
        df_lots = df_lots.drop(mixed)

        df_lots = df_lots.dropna().reset_index(drop=True)
        df_lots = df_lots[df_lots["lot_target"].str.endswith("01")]
        df_lots["lot_reacted"] = df_lots["lot_reacted"].apply(extract_lot)
        return df_lots.reset_index(drop=True)

    def _attach_reacted(self, df_lots: pd.DataFrame) -> pd.DataFrame:
        df = df_lots.copy()
        react_init = self.data["반응_init"]
        react_step = self.data["반응_step"]

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

        for i, row in df.iterrows():
            lot = row["lot_reacted"]

            init_match = react_init[
                react_init["생산LOT번호"].str.upper() == lot.upper()
            ]
            if init_match.empty:
                self.error_log.append(f"반응_init 매칭 실패: {lot}")
                _debug_log(self.debug, f"반응_init 매칭 실패: {lot}")
                continue
            df.at[i, "df_react_init"] = init_match.iloc[0].to_dict()

            step_match = react_step[
                react_step["생산LOT번호"].str.upper() == lot.upper()
            ]
            if step_match.empty:
                self.error_log.append(f"반응_step 매칭 실패: {lot}")
                _debug_log(self.debug, f"반응_step 매칭 실패: {lot}")
                continue

            self._parse_steps(df, i, step_match, lot)

        return df

    def _parse_steps(self, df: pd.DataFrame, i: int,
                     step_df: pd.DataFrame, lot: str) -> None:
        def to_float_list(col):
            return step_df[col].astype(str).str.replace(",", "").astype(float).tolist()

        weights_metal = to_float_list("Unnamed: 5")
        weights_naoh = to_float_list("Unnamed: 8")
        weights_nh4oh = to_float_list("Unnamed: 11")
        list_ph = to_float_list("PH")
        list_rpm = to_float_list("교반기RPM")

        if 0.0 in weights_metal + weights_naoh + weights_nh4oh + list_ph + list_rpm:
            self.error_log.append(f"투입량에 0 포함: {lot}")
            _debug_log(self.debug, f"투입량에 0 포함: {lot}")
            return

        df.at[i, "weights_metal"] = weights_metal
        df.at[i, "weights_naoh"] = weights_naoh
        df.at[i, "weights_nh4oh"] = weights_nh4oh
        df.at[i, "steps_ph"] = list_ph
        df.at[i, "steps_rpm"] = list_rpm

        step_nums = step_df["투입STEP"].astype(int).tolist()
        if step_nums[0] != 1:
            self.error_log.append(f"첫 STEP이 1이 아님: {lot}")
            return

        df.at[i, "steps_num"] = step_nums
        df.at[i, "steps_time"] = step_df["투입시간"].astype(float).tolist()

        lots_metal = step_df["Unnamed: 6"].dropna().unique().tolist()
        lots_naoh = step_df["Unnamed: 9"].dropna().unique().tolist()
        lots_nh4oh = step_df["Unnamed: 12"].dropna().unique().tolist()

        if not lots_metal or not lots_naoh or not lots_nh4oh:
            self.error_log.append(f"LOT 번호 비어 있음: {lot}")
            return

        df.at[i, "lots_metal"] = lots_metal
        df.at[i, "lots_naoh"] = lots_naoh
        df.at[i, "lots_nh4oh"] = lots_nh4oh

        dict_metal: dict = {"lot_metal": [], "weight": {}}
        dict_naoh: dict = {"lot_naoh": [], "weight": {}}
        dict_nh4oh: dict = {"lot_nh4oh": [], "weight": {}}

        for k, (_, step_row) in enumerate(step_df.iterrows()):
            lot_m = step_row["Unnamed: 6"]
            lot_n = step_row["Unnamed: 9"]
            lot_nh = step_row["Unnamed: 12"]

            w_m = float(str(step_row["Unnamed: 5"]).replace(",", ""))
            w_n = float(str(step_row["Unnamed: 8"]).replace(",", ""))
            w_nh = float(str(step_row["Unnamed: 11"]).replace(",", ""))

            for lot_val, w_val, d in [(lot_m, w_m, dict_metal),
                                       (lot_n, w_n, dict_naoh),
                                       (lot_nh, w_nh, dict_nh4oh)]:
                key = "lot_metal" if d is dict_metal else (
                    "lot_naoh" if d is dict_naoh else "lot_nh4oh")
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

    def _attach_handrecorded(self, df: pd.DataFrame) -> pd.DataFrame:
        if "df_handrecorded" not in df.columns:
            df["df_handrecorded"] = None

        hand_data = self.data.get("수기운전일지", {})
        for i, row in df.iterrows():
            lot = row["lot_reacted"]
            matched = False
            for line, df_list in hand_data.items():
                for df_hand in df_list:
                    try:
                        lot_hand = remove_trailing_non_digits(df_hand.iloc[2, 2])
                    except (IndexError, TypeError):
                        continue
                    if lot_hand == lot:
                        df.at[i, "df_handrecorded"] = df_hand
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                self.error_log.append(f"수기운전일지 없음: {lot}")
                _debug_log(self.debug, f"수기운전일지 없음: {lot}")

        return df

    def _attach_melting(self, df: pd.DataFrame) -> pd.DataFrame:
        melt_cols = ["df_melted", "lots_coso4", "lots_mnso4", "lots_niso4",
                     "weight_coso4", "weight_mnso4", "weight_niso4"]
        for col in melt_cols:
            df[col] = None

        melt_metal = self.data.get("용해_metal", pd.DataFrame())
        if melt_metal.empty:
            return df

        for i, row in df.iterrows():
            lots_metal = row["lots_metal"]
            if lots_metal is None:
                continue

            coso4_lots, mnso4_lots, niso4_lots = [], [], []
            coso4_ws, mnso4_ws, niso4_ws = [], [], []

            for lot_m in lots_metal:
                matched = melt_metal[
                    melt_metal["반응생산 LOT 번호"].str.upper() == row["lot_reacted"].upper()
                ]
                if matched.empty:
                    continue

                for mat, lst_l, lst_w in [
                    ("황산코발트", coso4_lots, coso4_ws),
                    ("황산망간", mnso4_lots, mnso4_ws),
                    ("황산니켈", niso4_lots, niso4_ws),
                ]:
                    mat_rows = matched[matched["원료명"] == mat]
                    lst_l.append(mat_rows["LOT NO"].tolist())
                    lst_w.append(
                        mat_rows["투입중량"].astype(str).str.replace(",", "")
                        .astype(float).tolist()
                    )

            if coso4_lots:
                df.at[i, "lots_coso4"] = coso4_lots
                df.at[i, "lots_mnso4"] = mnso4_lots
                df.at[i, "lots_niso4"] = niso4_lots
                df.at[i, "weight_coso4"] = coso4_ws
                df.at[i, "weight_mnso4"] = mnso4_ws
                df.at[i, "weight_niso4"] = niso4_ws

        return df

    def _attach_materials(self, df: pd.DataFrame) -> pd.DataFrame:
        mat_data = self.data.get("원재료", {})

        df["df_naoh"] = None
        for i, row in df.iterrows():
            lots = row["lots_naoh"]
            if lots is None:
                continue
            naoh_infos = []
            naoh_df = mat_data.get("NAOH", pd.DataFrame())
            if naoh_df.empty:
                continue
            for lot in lots:
                lot_key = lot.split("-")[0]
                match = naoh_df[naoh_df["입고LOT"].str.startswith(lot_key)]
                if not match.empty:
                    naoh_infos.append(match.iloc[0].to_dict())
            if len(naoh_infos) == len(lots):
                df.at[i, "df_naoh"] = naoh_infos

        for mat_col, mat_key, lots_col in [
            ("df_coso4", "COSO4", "lots_coso4"),
            ("df_mnso4", "MNSO4", "lots_mnso4"),
            ("df_niso4", "NISO4", "lots_niso4"),
        ]:
            df[mat_col] = None
            mat_df = mat_data.get(mat_key, pd.DataFrame())
            if mat_df.empty:
                continue

            for i, row in df.iterrows():
                lot_groups = row.get(lots_col)
                if lot_groups is None:
                    continue
                all_infos = []
                for group in lot_groups:
                    group_infos = []
                    for lot in group:
                        lot_key = lot.split("-")[0]
                        match = mat_df[mat_df["입고LOT"].str.startswith(lot_key)]
                        if not match.empty:
                            group_infos.append(match.iloc[0].to_dict())
                    if len(group_infos) == len(group):
                        all_infos.append(group_infos)
                if len(all_infos) == len(lot_groups):
                    df.at[i, mat_col] = all_infos

        return df
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_tracker.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/data/tracker.py tests/test_tracker.py tests/conftest.py
git commit -m "feat: rewrite tracker for consolidated data, LOT-based matching"
```

---

### Task 4: Update preprocessor.py (initial conditions + weighted sum)

**Files:**
- Modify: `src/data/preprocessor.py`
- Modify: `tests/test_preprocessor.py`

- [ ] **Step 1: Write tests for updated preprocessor**

Replace `tests/test_preprocessor.py`:

```python
import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocesser


def test_preprocessor_wide_format(dummy_data, seq_1_6_path):
    from src.data.tracker import TrackerRawData
    tracker = TrackerRawData(dummy_data, ["1라인"], "N86L")
    prep = DataPreprocesser(
        df_tracked=tracker.df_tracked,
        seq_path=seq_1_6_path,
        is_material=False,
        is_handrecorded=True,
    )
    df = prep.df_preprocessed
    assert isinstance(df, pd.DataFrame)
    assert "lot_target" in df.columns
    assert "lot_reacted" in df.columns
    assert "반응투입_초기조건_pH" in df.columns
    assert "STEP_WEIGHT_Metal_01" in df.columns


def test_preprocessor_no_impellar(dummy_data, seq_1_6_path):
    from src.data.tracker import TrackerRawData
    tracker = TrackerRawData(dummy_data, ["1라인"], "N86L")
    prep = DataPreprocesser(
        df_tracked=tracker.df_tracked,
        seq_path=seq_1_6_path,
        is_material=False,
        is_handrecorded=False,
    )
    df = prep.df_preprocessed
    impellar_cols = [c for c in df.columns if "IMPELLAR" in c.upper()]
    assert len(impellar_cols) == 0


def test_preprocessor_init_has_9_columns(dummy_data, seq_1_6_path):
    from src.data.tracker import TrackerRawData
    tracker = TrackerRawData(dummy_data, ["1라인"], "N86L")
    prep = DataPreprocesser(
        df_tracked=tracker.df_tracked,
        seq_path=seq_1_6_path,
        is_material=False,
        is_handrecorded=False,
    )
    df = prep.df_preprocessed
    init_cols = [c for c in df.columns if c.startswith("반응투입_초기조건_")]
    assert len(init_cols) == 9


def test_preprocessor_step_size_columns(dummy_data, seq_1_6_path):
    from src.data.tracker import TrackerRawData
    tracker = TrackerRawData(dummy_data, ["1라인"], "N86L")
    prep = DataPreprocesser(
        df_tracked=tracker.df_tracked,
        seq_path=seq_1_6_path,
        is_material=False,
        is_handrecorded=True,
    )
    df = prep.df_preprocessed
    size_cols = [c for c in df.columns if "STEP_SIZE_D50" in c]
    assert len(size_cols) > 0


def test_preprocessor_material_weighted_sum(dummy_data, seq_1_6_path):
    from src.data.tracker import TrackerRawData
    tracker = TrackerRawData(dummy_data, ["1라인"], "N86L")
    prep = DataPreprocesser(
        df_tracked=tracker.df_tracked,
        seq_path=seq_1_6_path,
        is_material=True,
        is_handrecorded=False,
    )
    df = prep.df_preprocessed
    mat_cols = [c for c in df.columns if c.startswith("STEP_MATERIAL_")]
    assert len(mat_cols) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocessor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement updated preprocessor.py**

Replace `src/data/preprocessor.py` entirely:

```python
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

HAND_COL_INDICES = [0, 2, 3, 4, 5, 6]
HAND_COL_NAMES = ["time", "dmin", "d10", "d50", "d90", "dmax"]

_STEP_PREFIXES = ["STEP_WEIGHT_Metal_", "STEP_WEIGHT_NaOH_", "STEP_WEIGHT_NH4OH_",
                  "STEP_PH_", "STEP_RPM_"]
_SIZE_PREFIXES = ["STEP_SIZE_DMIN_", "STEP_SIZE_D10_", "STEP_SIZE_D50_",
                  "STEP_SIZE_D90_", "STEP_SIZE_DMAX_"]

MATERIAL_FEATURES = [
    ("황산망간_성분_ChemicalComposition(C01)_Mn", "df_mnso4", "Chemical Composition (C01)_Mn"),
    ("황산망간_성분_InitialPH(C03)_pH", "df_mnso4", "Initial PH (C03)_pH"),
    ("황산니켈_성분_ChemicalComposition(C01)_Ni", "df_niso4", "Chemical Composition (C01)_Ni"),
    ("가성소다_성분_ChemicalComposition(C01)_NaOH", "df_naoh", "Chemical Composition (C01)_NaOH"),
    ("황산코발트_투입량", "weight_coso4", None),
]


def _load_seq(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["idx", "time"])


def _step_col(prefix: str, j: int) -> str:
    return f"{prefix}{str(j + 1).zfill(2)}"


def _find_col_case_insensitive(d: dict, target: str) -> tp.Any:
    target_lower = target.lower()
    for k, v in d.items():
        if str(k).lower() == target_lower:
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
        df_raw = self.df_tracked.copy()
        df_raw = self._filter(df_raw)
        df_pre = self._build_features(df_raw)
        return df_pre, df_raw

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_material:
            non_step_cols = [c for c in df.columns
                             if not any(t in c for t in [
                                 "df_handrecorded", "step_info", "df_coso4"])]
            df = df.dropna(subset=non_step_cols)
        else:
            df = df.dropna(subset=["df_react_init"])

        if self.is_handrecorded:
            df = df.dropna(subset=["df_handrecorded", "steps_num"])

        return df.reset_index(drop=True)

    def _build_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        n = len(df_raw)

        init_data = {col: np.full(n, np.nan) for col in INIT_COLUMNS}
        for i, (_, row) in enumerate(df_raw.iterrows()):
            init_dict = row.get("df_react_init")
            if init_dict is None:
                continue
            for feat_col, src_col in INIT_COL_MAP.items():
                val = _find_col_case_insensitive(init_dict, src_col)
                if val is not None:
                    try:
                        init_data[feat_col][i] = float(
                            str(val).replace(",", ""))
                    except (ValueError, TypeError):
                        pass

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

        df = pd.concat([
            df_raw[["lot_target", "lot_reacted"]].reset_index(drop=True),
            pd.DataFrame(init_data),
            pd.DataFrame(step_data),
        ], axis=1)

        if self.is_material:
            df = self._build_material_features(df, df_raw)

        if self.is_handrecorded:
            df = self._interpolate_handrecorded(df, df_raw)

        return df

    def _build_material_features(
        self, df: pd.DataFrame, df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        n = len(df_raw)
        mat_data: tp.Dict[str, np.ndarray] = {}

        for feat_suffix, _, _ in MATERIAL_FEATURES:
            for j in range(INT_LENGTH):
                col = _step_col(f"STEP_MATERIAL_{str(j+1).zfill(2)}_", 0).replace(
                    "_01", f"_{feat_suffix}")
                real_col = f"STEP_MATERIAL_{str(j+1).zfill(2)}_{feat_suffix}"
                mat_data[real_col] = np.full(n, np.nan)

        for i, (_, row) in enumerate(df_raw.iterrows()):
            for j in range(INT_LENGTH):
                step_info = row.get(f"step_info_{str(j+1).zfill(2)}")
                if step_info is None:
                    break

                for feat_suffix, src_col, prop_key in MATERIAL_FEATURES:
                    out_col = f"STEP_MATERIAL_{str(j+1).zfill(2)}_{feat_suffix}"

                    if prop_key is None and src_col == "weight_coso4":
                        weights = row.get("weight_coso4")
                        if weights:
                            total = sum(sum(g) for g in weights)
                            mat_data[out_col][i] = total
                        continue

                    mat_infos = row.get(src_col)
                    if mat_infos is None:
                        continue

                    if src_col == "df_naoh":
                        info_key = "naoh"
                        lot_key = "lot_naoh"
                    elif src_col == "df_mnso4":
                        info_key = "metal"
                        lot_key = "lot_metal"
                    elif src_col == "df_niso4":
                        info_key = "metal"
                        lot_key = "lot_metal"
                    else:
                        continue

                    step_lots = step_info.get(info_key, {})
                    lot_names = step_lots.get(lot_key, [])
                    lot_weights = step_lots.get("weight", {})

                    total_w = 0.0
                    weighted_sum = 0.0

                    if src_col == "df_naoh":
                        for info in mat_infos:
                            lot_name = info.get("입고LOT", "")
                            cum_w = sum(lot_weights.get(
                                next((l for l in lot_names
                                      if l.split("-")[0] == lot_name.split("-")[0]),
                                     ""), [0]))
                            prop_val = info.get(prop_key, np.nan)
                            if not np.isnan(prop_val) and cum_w > 0:
                                weighted_sum += cum_w * prop_val
                                total_w += cum_w
                    else:
                        for group_infos in mat_infos:
                            for info in group_infos:
                                lot_name = info.get("입고LOT", "")
                                prop_val = info.get(prop_key, np.nan)
                                if not np.isnan(prop_val):
                                    weighted_sum += prop_val
                                    total_w += 1

                    if total_w > 0:
                        mat_data[out_col][i] = weighted_sum / total_w

        df = pd.concat([df, pd.DataFrame(mat_data, index=df.index)], axis=1)
        return df

    def _interpolate_handrecorded(
        self, df: pd.DataFrame, df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        n = len(df_raw)
        cumtimes = np.cumsum(self.df_seq["time"].tolist())

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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preprocessor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: update preprocessor for new data format, add weighted sum"
```

---

### Task 5: Update preprocess.py CLI

**Files:**
- Modify: `preprocess.py`

- [ ] **Step 1: Update preprocess.py for new API**

```python
"""
전처리 실행 스크립트.
Usage:
    python preprocess.py --data-path ../../data/new_structure
                         --product n86l
                         --output-dir ../../data/preprocessed
                         [--no-material] [--no-handrecorded] [--debug]
"""
import os
import argparse
import pandas as pd

from src.config.schema import PRODUCT_N86L, PRODUCT_N86S
from src.data.loader import get_alldata
from src.data.tracker import TrackerRawData
from src.data.preprocessor import DataPreprocesser


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--product", required=True, choices=["n86l", "n86s"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-material", action="store_true")
    p.add_argument("--no-handrecorded", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PRODUCT_N86L if args.product == "n86l" else PRODUCT_N86S
    is_material = not args.no_material
    is_handrecorded = not args.no_handrecorded

    print(f"Loading data from: {args.data_path}")
    data = get_alldata(args.data_path, debug=args.debug)

    print("Tracking LOTs...")
    tracker = TrackerRawData(data, cfg.lines, cfg.name, debug=args.debug)

    if tracker.error_log:
        print(f"  {len(tracker.error_log)} errors logged:")
        for msg in tracker.error_log[:10]:
            print(f"    {msg}")

    print("Preprocessing features...")
    preprocesser = DataPreprocesser(
        df_tracked=tracker.df_tracked,
        seq_path=cfg.seq_path,
        is_material=is_material,
        is_handrecorded=is_handrecorded,
    )

    df_pre = preprocesser.df_preprocessed
    df_raw = preprocesser.df_filtered

    print(f"  {len(df_pre)} batches total")

    suffix = f"{'m' if is_material else 'mn'}_{'h' if is_handrecorded else 'hn'}"
    folder_name = f"공침데이터_{cfg.name}_{suffix}"
    out_path = os.path.join(args.output_dir, folder_name)
    os.makedirs(out_path, exist_ok=True)

    df_pre.to_pickle(os.path.join(out_path, "data_preprocessed.pkl"))
    df_raw.to_pickle(os.path.join(out_path, "data_raw.pkl"))

    with open(os.path.join(out_path, "config.txt"), "w") as f:
        f.write(f"is_material={is_material}\n")
        f.write(f"is_handrecorded={is_handrecorded}\n")
        f.write(f"product={cfg.name}\n")

    print(f"Saved to: {out_path}")
    print(f"  data_preprocessed.pkl: {len(df_pre)} rows, {len(df_pre.columns)} cols")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `python preprocess.py --help`
Expected: `--debug` flag shown in help output

- [ ] **Step 3: Commit**

```bash
git add preprocess.py
git commit -m "feat: update CLI for new data pipeline, add --debug flag"
```

---

### Task 6: Run full test suite and fix issues

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Fix any remaining failures**

Address test failures from integration between updated modules. Common issues:
- Old fixture references in test files
- Import path changes
- Column name mismatches

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "fix: resolve integration test issues"
```
