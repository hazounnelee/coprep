import re

import pytest
import pandas as pd
import numpy as np


def make_dummy_통합일지(n_lots: int = 3, product: str = "N86L") -> pd.DataFrame:
    """통합일지 Sheet1 모사. 4 header rows + data rows.

    Columns: No., 포장 Lot No.
    Rows 0-3 are header/metadata (None), row 4+ is data.
    """
    rows = []
    for _ in range(4):
        rows.append({col: None for col in ["No.", "포장 Lot No"]})
    for i in range(n_lots):
        lot_reacted = f"{product}-1A25050{i+1}-02"
        lot_target = f"PKG-{i+1:04d}-01"
        rows.append({"No.": lot_reacted, "포장 Lot No": lot_target})
    return pd.DataFrame(rows)


def make_dummy_수기운전일지(lot: str, n_measurements: int = 10) -> pd.DataFrame:
    """수기운전일지 모사. row2 col2 = lot, row12~ = 측정 데이터."""
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
        row = [str(t), None, str(d50 * 0.8), str(d50 * 0.9), str(d50),
               str(d50 * 1.1), str(d50 * 1.2)]
        rows.append(row)
    return pd.DataFrame(rows)


def make_dummy_반응_init(lots: list) -> pd.DataFrame:
    """반응_init DataFrame 모사 (반응투입_LOT_INIT.xlsx 결과)."""
    rows = []
    for lot in lots:
        rows.append({
            "생산LOT번호": lot,
            "초기시작일시": "2025-05-01 08:00:00",
            "초기종료일시": "2025-05-01 16:00:00",
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
    """반응_step DataFrame 모사 (반응투입_LOT_STEP.xlsx 결과).

    Column mapping from task spec:
      Unnamed: 5 -> Metal 투입량(kg)
      Unnamed: 6 -> Metal 투입LOT번호
      Unnamed: 8 -> NaOH 투입량(kg)
      Unnamed: 9 -> NaOH 투입LOT번호
      Unnamed: 11 -> NH4OH 투입량(kg)
      Unnamed: 12 -> NH4OH 투입LOT번호
      PH -> pH
      교반기RPM -> RPM
    """
    rows = []
    np.random.seed(99)
    for lot in lots:
        for step in range(1, n_steps + 1):
            rows.append({
                "생산LOT번호": lot,
                "투입STEP": step,
                "투입시간": 60.0,
                "Unnamed: 3": str(step * 60),
                "Metal Solution": "Metal Sol.",
                "Unnamed: 5": f"{150.0 + np.random.randn() * 5:.1f}",
                "Unnamed: 6": "METAL-LOT-001",
                "NAOH": "NaOH",
                "Unnamed: 8": f"{80.0 + np.random.randn() * 2:.1f}",
                "Unnamed: 9": "NAOH-LOT-001",
                "NH4OH": "NH4OH",
                "Unnamed: 11": f"{50.0 + np.random.randn() * 1:.1f}",
                "Unnamed: 12": "NH4OH-LOT-001",
                "PH": f"{11.5 + np.random.randn() * 0.1:.2f}",
                "교반기RPM": f"{700 + np.random.randint(-10, 10)}",
            })
    return pd.DataFrame(rows)


def make_dummy_용해_metal(lots_metal: list, reaction_lots: list) -> pd.DataFrame:
    """용해_metal DataFrame 모사 (MELT_WRK_ORD_METAL.xlsx 결과).

    Each reaction_lot is linked to a metal lot.
    Each metal lot has rows for 황산코발트, 황산망간, 황산니켈.
    """
    rows = []
    for lot_m, lot_r in zip(lots_metal, reaction_lots):
        for mat, lot_no, weight in [
            ("황산코발트", "COSO4-001-01", "100.0"),
            ("황산망간", "MNSO4-001-01", "200.0"),
            ("황산니켈", "NISO4-001-01", "500.0"),
        ]:
            rows.append({
                "생산Lot번호": lot_m,
                "원료명": mat,
                "LOT NO": lot_no,
                "투입중량": weight,
                "반응생산 LOT 번호": lot_r,
            })
    return pd.DataFrame(rows)


def make_dummy_원재료() -> dict:
    """원재료 dict of DataFrames by material type.

    Matches loader output: {"COSO4": df, "MNSO4": df, "NISO4": df, "NAOH": df, "NH4OH": df}
    Each DataFrame has 입고LOT + property columns matching MATERIAL_PROPS.
    """
    result = {}

    # COSO4: 입고LOT, Co
    result["COSO4"] = pd.DataFrame({
        "입고LOT": ["COSO4-001-01", "COSO4-002-01"],
        "Co": [0.235, 0.240],
    })

    # MNSO4: 입고LOT, Mn, pH
    result["MNSO4"] = pd.DataFrame({
        "입고LOT": ["MNSO4-001-01", "MNSO4-002-01"],
        "Mn": [0.310, 0.315],
        "pH": [3.5, 3.6],
    })

    # NISO4: 입고LOT, Ni
    result["NISO4"] = pd.DataFrame({
        "입고LOT": ["NISO4-001-01", "NISO4-002-01"],
        "Ni": [0.220, 0.225],
    })

    # NAOH: 입고LOT, NaOH
    result["NAOH"] = pd.DataFrame({
        "입고LOT": ["NAOH-LOT-001", "NAOH-LOT-002"],
        "NaOH": [45.0, 46.0],
    })

    # NH4OH: 입고LOT only
    result["NH4OH"] = pd.DataFrame({
        "입고LOT": ["NH4OH-LOT-001", "NH4OH-LOT-002"],
    })

    return result


@pytest.fixture
def dummy_data():
    """New-format data dict matching loader.get_alldata() output."""
    np.random.seed(42)

    lots = ["N86L-1A250501-02", "N86L-1A250502-02", "N86L-1A250503-02"]
    metal_lots = ["METAL-LOT-001", "METAL-LOT-002", "METAL-LOT-003"]

    data = {}

    # 통합일지: {line_key: DataFrame}
    data["통합일지"] = {"1라인": make_dummy_통합일지(n_lots=3)}

    # 수기운전일지: {line_key: [DataFrame, ...]}
    data["수기운전일지"] = {
        "1라인": [make_dummy_수기운전일지(lot) for lot in lots],
    }

    # 반응_init: single DataFrame with all LOTs
    data["반응_init"] = make_dummy_반응_init(lots)

    # 반응_step: single DataFrame with all LOTs x steps
    data["반응_step"] = make_dummy_반응_step(lots, n_steps=20)

    # 용해_metal: single DataFrame linking metal lots to reaction lots
    data["용해_metal"] = make_dummy_용해_metal(metal_lots, lots)

    # 용해_ord: empty for now (not used by tracker directly)
    data["용해_ord"] = pd.DataFrame()

    # 원재료: dict of DataFrames by type
    data["원재료"] = make_dummy_원재료()

    return data


@pytest.fixture
def seq_1_6_path(tmp_path):
    p = tmp_path / "seq_1_6.txt"
    lines = [(i + 1, 60 if i not in [4, 5, 6, 7, 10, 11, 21, 22] else 30)
             for i in range(53)]
    p.write_text("\n".join(f"{s}\t{t}" for s, t in lines))
    return str(p)
