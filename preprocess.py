"""
전처리 실행 스크립트.
Usage:
    python preprocess.py --data-path ../../data/non_image_original/전구체/공침데이터_대입경
                         --product n86l
                         --output-dir ../../data/non_image_organized/전구체
                         [--no-material] [--no-handrecorded]
"""
import os
import argparse
import pickle
import pandas as pd
from collections import defaultdict

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
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PRODUCT_N86L if args.product == "n86l" else PRODUCT_N86S
    is_material = not args.no_material
    is_handrecorded = not args.no_handrecorded

    print(f"Loading Excel from: {args.data_path}")
    data_excel = get_alldata(args.data_path)

    print("Tracking LOTs...")
    tracker = TrackerRawData(data_excel, cfg.lines, cfg.name)

    if tracker.error_log:
        print(f"  {len(tracker.error_log)} errors logged:")
        for msg in tracker.error_log[:10]:
            print(f"    {msg}")

    print("Preprocessing features...")
    preprocesser = DataPreprocesser(
        dict_raw=tracker.dict_lines_tracked,
        list_lines=cfg.lines,
        seq_path=cfg.seq_path,
        is_material=is_material,
        is_handrecorded=is_handrecorded,
    )

    list_pre = [preprocesser.dict_lines_preprocessed[l] for l in cfg.lines]
    list_raw = [preprocesser.dict_df_filtered[l] for l in cfg.lines]

    df_pre = pd.concat(list_pre, ignore_index=True)
    df_raw = pd.concat(list_raw, ignore_index=True)

    for line in cfg.lines:
        n = len(preprocesser.dict_lines_preprocessed[line])
        print(f"  {line}: {n}개 배치")

    # 저장
    suffix = f"{'m' if is_material else 'mn'}_{'h' if is_handrecorded else 'hn'}"
    folder_name = f"공침데이터_{cfg.name}_{suffix}"
    out_path = os.path.join(args.output_dir, folder_name)
    os.makedirs(out_path, exist_ok=True)

    df_pre.to_pickle(os.path.join(out_path, "data_preprocessed.pkl"))
    df_raw.to_pickle(os.path.join(out_path, "data_raw.pkl"))

    # is_material, is_handrecorded 상태 저장
    with open(os.path.join(out_path, "config.txt"), "w") as f:
        f.write(f"is_material={is_material}\n")
        f.write(f"is_handrecorded={is_handrecorded}\n")
        f.write(f"product={cfg.name}\n")

    print(f"Saved to: {out_path}")
    print(f"  data_preprocessed.pkl: {len(df_pre)} rows, {len(df_pre.columns)} cols")


if __name__ == "__main__":
    main()
