"""
진단 스크립트. 아래처럼 실행:
  python diagnose.py --data-path ../../data/new_structure --product n86l
"""
import argparse
from src.data.loader import get_alldata
from src.data.tracker import TrackerRawData, PATTERN_LOT_LINE
from src.config.schema import PRODUCT_N86L, PRODUCT_N86S

p = argparse.ArgumentParser()
p.add_argument("--data-path", required=True)
p.add_argument("--product", required=True, choices=["n86l", "n86s"])
args = p.parse_args()

cfg = PRODUCT_N86L if args.product == "n86l" else PRODUCT_N86S

print("=== 1. 데이터 로드 ===")
data = get_alldata(args.data_path, debug=True)

df_init = data["반응_init"]
print(f"\n=== 2. 반응_init 상태 ===")
print(f"empty: {df_init.empty}")
print(f"rows: {len(df_init)}")
print(f"cols: {df_init.columns.tolist()}")
print(f"'생산LOT번호' in cols: {'생산LOT번호' in df_init.columns}")

print(f"\n=== 3. product_name 필터 ===")
lots = df_init["생산LOT번호"].dropna().astype(str).unique().tolist() if "생산LOT번호" in df_init.columns else []
print(f"전체 LOT 수: {len(lots)}")
print(f"샘플 LOT (앞 5개): {lots[:5]}")

lots_filtered = [lot for lot in lots if cfg.name.lower() in lot.lower()]
print(f"product_name='{cfg.name}' 필터 후: {len(lots_filtered)}개")

print(f"\n=== 4. 라인 번호 정규식 매칭 ===")
for lot in lots_filtered[:10]:
    m = PATTERN_LOT_LINE.match(lot)
    result = f"{m.group(1)}라인 (매칭 성공)" if m else "매칭 실패"
    print(f"  {lot} -> {result}")

print(f"\n=== 5. TrackerRawData 실행 ===")
tracker = TrackerRawData(data, cfg.lines, cfg.name, debug=True, lot_source="반응투입")
print(f"df_tracked shape: {tracker.df_tracked.shape}")
if not tracker.df_tracked.empty:
    print(f"df_tracked columns (앞 10개): {tracker.df_tracked.columns.tolist()[:10]}")
print(f"error_log 수: {len(tracker.error_log)}")
for msg in tracker.error_log[:5]:
    print(f"  {msg}")
