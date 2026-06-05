import os
import time
import typing as tp
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, root_mean_squared_error

from src.config.schema import XGBConfig, DataSettings
from src.model.postprocessor import split_xy, augment_data, filter_material, postprocess_by_product

ZERO_THRESHOLD = 0.01


def load_preprocessed(path: str) -> tp.Tuple[pd.DataFrame, bool, bool]:
    df = pd.read_pickle(os.path.join(path, "data_preprocessed.pkl"))
    config = {}
    with open(os.path.join(path, "config.txt")) as f:
        for line in f:
            k, v = line.strip().split("=")
            config[k] = v == "True"
    return df, config["is_material"], config["is_handrecorded"]


def train(
    df: pd.DataFrame,
    product_type: str,
    xgb_cfg: XGBConfig,
    data_settings: DataSettings,
    output_dir: str,
) -> dict:
    # 이상치 제거 (Metal > 4000)
    metal_cols = [c for c in df.columns if "STEP_WEIGHT_Metal" in c]
    df = df[~(df[metal_cols] > 4000).any(axis=1)]

    if data_settings.is_material_filtered:
        df = filter_material(df)

    # LOT 기준 split
    if data_settings.is_splitlot:
        gss = GroupShuffleSplit(1, test_size=data_settings.size_test,
                                random_state=xgb_cfg.num_seed)
        for train_idx, test_idx in gss.split(df, groups=df["lot_reacted"]):
            data_train = df.iloc[train_idx]
            data_test = df.iloc[test_idx]
    else:
        data_train, data_test = train_test_split(
            df, test_size=data_settings.size_test, random_state=xgb_cfg.num_seed
        )

    # Augmentation
    if data_settings.is_augment:
        data_train = augment_data(data_train, data_settings.is_material_filtered)
        data_test = augment_data(data_test, data_settings.is_material_filtered)
        x_train, y_train = split_xy(data_train)
        x_test, y_test = split_xy(data_test)
    else:
        data_train = data_train.fillna(0.0)
        data_test = data_test.fillna(0.0)
        x_train, y_train = split_xy(data_train)
        x_test, y_test = split_xy(data_test)
        x_train = x_train.apply(pd.to_numeric, errors="coerce")
        y_train = y_train.apply(pd.to_numeric, errors="coerce")
        x_test = x_test.apply(pd.to_numeric, errors="coerce")
        y_test = y_test.apply(pd.to_numeric, errors="coerce")

    x_train, y_train = postprocess_by_product(x_train, y_train, product_type)
    x_test, y_test = postprocess_by_product(x_test, y_test, product_type)

    x_train = x_train.drop(["lot_target", "lot_reacted"], axis=1, errors="ignore")
    x_test = x_test.drop(["lot_target", "lot_reacted"], axis=1, errors="ignore")

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=xgb_cfg.num_seed
    )

    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    model = XGBRegressor(
        n_estimators=xgb_cfg.num_round,
        max_depth=xgb_cfg.max_depth,
        learning_rate=xgb_cfg.eta,
        objective=xgb_cfg.objective,
        random_state=xgb_cfg.num_seed,
        enable_categorical=xgb_cfg.enable_categorical,
        tree_method=xgb_cfg.tree_method,
        early_stopping_rounds=xgb_cfg.early_stopping_rounds,
    )

    t0 = time.time()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=50)
    print(f"Training time: {time.time() - t0:.1f}s")

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # 실존 Y 기준 평가: y_true가 NaN인 스텝(측정 없음)은 평가에서 제외
    y_true_df = y_test.copy()
    y_true_df.mask(y_true_df.abs() < ZERO_THRESHOLD, inplace=True)
    y_pred_df = pd.DataFrame(y_pred, columns=y_true_df.columns, index=y_true_df.index)
    # Step 1: 예측값이 임계값 미만이면 0으로 패널티 (y_true가 존재하는 셀에만 유효)
    y_pred_df.mask(y_pred_df.abs() < ZERO_THRESHOLD, 0.0, inplace=True)
    # Step 2: y_true가 NaN인 위치는 평가 제외 (step 1의 0.0 덮어씌움 — 의도적)
    y_pred_df.mask(y_true_df.isna(), inplace=True)
    y_pred_df.fillna(0.0, inplace=True)

    valid = np.isfinite(y_true_df.values.flatten())
    r2_real = r2_score(y_true_df.values.flatten()[valid], y_pred_df.values.flatten()[valid])
    rmse_real = root_mean_squared_error(
        y_true_df.values.flatten()[valid], y_pred_df.values.flatten()[valid]
    )

    results = {"R2": r2, "RMSE": rmse, "R2_real": r2_real, "RMSE_real": rmse_real}

    os.makedirs(output_dir, exist_ok=True)
    model.save_model(os.path.join(output_dir, "model_xgb.json"))
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}={v:.4f}\n")

    return results
