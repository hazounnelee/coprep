"""
학습 실행 스크립트.
Usage:
    python train.py --data-path ../../data/non_image_organized/전구체/공침데이터_N86L_m_h
                    --product-type 대입경
                    --output-dir ./results/exp_001
                    [--no-augment] [--no-splitlot]
"""
import argparse
from src.config.schema import XGBConfig, DataSettings
from src.model.trainer import load_preprocessed, train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--product-type", required=True, choices=["대입경", "소입경"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-splitlot", action="store_true")
    p.add_argument("--no-material-filter", action="store_true")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()

    df, is_material, is_handrecorded = load_preprocessed(args.data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} cols")
    print(f"  preprocessed with: is_material={is_material}, is_handrecorded={is_handrecorded}")

    xgb_cfg = XGBConfig()
    data_settings = DataSettings(
        size_test=args.test_size,
        is_augment=not args.no_augment,
        is_material_filtered=not args.no_material_filter,
        is_splitlot=not args.no_splitlot,
    )

    results = train(df, args.product_type, xgb_cfg, data_settings, args.output_dir)

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
