import re

import pytest
import pandas as pd
import numpy as np
from collections import defaultdict


def make_dummy_통합일지(n_lots: int = 3, product: str = "N86L") -> pd.DataFrame:
    """통합일지 Sheet1 모사. No./포장 Lot No 컬럼 포함."""
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


def make_dummy_반응투입스케쥴실적(
    lot: str, date: str, facility: str, n_steps: int = 20
) -> dict:
    """반응투입스케쥴실적 Sheet1/2/3 모사."""
    sheet1 = pd.DataFrame({
        "작업시작일시": [f"20{date[:2]}-{date[2:4]}-{date[4:]} 00:00:00"],
        "설비명": [f"반응기 {facility}(GCOP01)"],
        "생산 LOT 번호": [lot],
        "No": ["▶"],
    })
    rows = [{}]
    for step in range(1, n_steps + 1):
        rows.append({
            "투입\nSTEP": step,
            "투입시간": 60.0,
            "Unnamed: 4": str(step * 60),
            "Unnamed: 6": f"{150.0 + np.random.randn() * 5:.1f}",
            "Unnamed: 7": "METAL-LOT-001",
            "Unnamed: 10": f"{80.0 + np.random.randn() * 2:.1f}",
            "Unnamed: 11": "NAOH-LOT-001",
            "Unnamed: 14": f"{50.0 + np.random.randn() * 1:.1f}",
            "Unnamed: 15": "NH4OH-LOT-001",
            "Unnamed: 19": "NH4OH-LOT-001",
            "PH": f"{11.5 + np.random.randn() * 0.1:.2f}",
            "교반기\nRPM": f"{700 + np.random.randint(-10, 10)}",
        })
    rows.append({})
    sheet2 = pd.DataFrame(rows)
    sheet3 = pd.DataFrame({
        "작업\n시간": [480],
        "순수\n온도": [50.0],
        "순수\n투입 중량(kg)": [1200.0],
        "NAOH\n투입중량(kg)": [200.0],
        "NH4OH\n투입중량(kg)": [100.0],
        "용존산소량": [0.5],
        "pH": [11.0],
        "N2\n주입 유량": [5.0],
        "N2\nPURGE 시간": [30.0],
        "IMPELLAR\n교반 속도": [700.0],
    })
    return {"Sheet1": sheet1, "Sheet2": sheet2, "Sheet3": sheet3}


def make_dummy_용해작업실적(lot_melted: str) -> dict:
    """용해작업실적 Sheet1/2 모사."""
    sheet1 = pd.DataFrame({
        "생산Lot번호": [lot_melted],
        "No": ["▶"],
    })
    sheet2 = pd.DataFrame({
        "원료명": ["황산니켈", "황산코발트", "황산망간"],
        "LOT NO": ["NISO4-001-01", "COSO4-001-01", "MNSO4-001-01"],
        "투입중량": ["500.0", "100.0", "200.0"],
    })
    return {"lot_melted": lot_melted, "df": {"Sheet1": sheet1, "Sheet2": sheet2}}


def make_dummy_원재료(lot_prefix: str, n_lots: int = 1) -> list:
    """원재료 검사 DataFrame 리스트 모사.

    lot_prefix must match what the tracker extracts via lot.split('-')[0].
    e.g. Sheet2 "NAOH-LOT-001" → prefix "NAOH" → 가성소다 입고LOT must contain "NAOH".
    """
    results = []
    for i in range(n_lots):
        data = {"입고LOT": [lot_prefix]}
        for j in range(17):
            data[f"prop_{j}"] = [str(round(np.random.uniform(0.1, 99.9), 3))]
        results.append(pd.DataFrame(data))
    return results


@pytest.fixture
def dummy_excel_data():
    """전체 data_excel DefaultDict 더미."""
    np.random.seed(42)
    data = defaultdict(lambda: defaultdict(list))

    lots = [
        ("N86L-1A250501-02", "PKG-0001-01"),
        ("N86L-1A250502-02", "PKG-0002-01"),
        ("N86L-1A250503-02", "PKG-0003-01"),
    ]

    data["1라인"]["통합일지"].append(make_dummy_통합일지(n_lots=3))

    for lot_reacted, _ in lots:
        data["1라인"]["수기운전일지"].append(make_dummy_수기운전일지(lot_reacted))

    for lot_reacted, _ in lots:
        parts = lot_reacted.split("-")
        m = re.match(r"(\d+)([A-Z]+)(\d{6})", parts[1])
        date, facility = m.group(3), m.group(2)
        # "df" wrapper required by tracker: d["df"]["Sheet1/2/3"]
        data["1라인"]["반응투입스케쥴실적"].append({
            "df": make_dummy_반응투입스케쥴실적(lot_reacted, date, facility),
            "date": f"20{date[:2]}-{date[2:4]}-{date[4:]} 00:00:00",
            "facility": f"반응기 {facility}(GCOP01)",
        })

    data["1라인"]["용해작업실적"].append(
        make_dummy_용해작업실적("METAL-LOT-001")
    )

    # lot_prefix must match lot.split("-")[0] from Sheet2 LOT NO columns:
    #   "NISO4-001-01" → "NISO4", "COSO4-001-01" → "COSO4", "MNSO4-001-01" → "MNSO4"
    #   "NAOH-LOT-001" → "NAOH"
    data["원재료"]["황산니켈"] = make_dummy_원재료("NISO4")
    data["원재료"]["황산코발트"] = make_dummy_원재료("COSO4")
    data["원재료"]["황산망간"] = make_dummy_원재료("MNSO4")
    data["원재료"]["가성소다"] = make_dummy_원재료("NAOH")
    data["원재료"]["암모니아"] = make_dummy_원재료("NH4OH")

    return data


@pytest.fixture
def seq_1_6_path(tmp_path):
    p = tmp_path / "seq_1_6.txt"
    lines = [(i + 1, 60 if i not in [4, 5, 6, 7, 10, 11, 21, 22] else 30)
             for i in range(53)]
    p.write_text("\n".join(f"{s}\t{t}" for s, t in lines))
    return str(p)
