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
