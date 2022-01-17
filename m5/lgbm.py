import pandas as pd
import lightgbm as lgb
from m5.utils import get_columns, move_column, create_dir
from m5.features import build_lag_features
from m5.config import LGBM_PARAMS
from m5.definitions import (
    TARGET, N_LAGS, ROOT_DIR, FH,
    AGG_LEVEL, CALENDAR_FEATURES, LAG_FEATURES)


def build_lags(data, target, step, lags):
    id_cols = get_columns(data, lambda x: x.endswith("id"))
    if not id_cols:
        dataset = build_lag_features(data, target, step, lags)
    else:
        dataset = data.groupby(id_cols, group_keys=False).apply(
            lambda df: build_lag_features(df, target, step, lags))
    return dataset


def prepare_dataset(target, lags, store, step):
    print(f"Preparing dataset for level {store} and step {step}")
    input_file = ROOT_DIR / f"data/processed/stores/{store}/data.parquet"
    output_dir = create_dir(ROOT_DIR / f"data/processed/datasets/{store}/{step}")
    data = pd.read_parquet(input_file)
    move_column(data, target)
    data.drop(columns=["dollar_sales"], inplace=True)
    dataset = build_lags(data, target, step, lags)

    N = dataset.d.max()
    train = dataset.loc[dataset.d <= (N - FH), :]
    val = dataset.loc[dataset.d > (N - FH), :]

    train.to_parquet(output_dir / "train.parquet")
    val.to_parquet(output_dir / "val.parquet")


def prepare_dataset_binaries(store, step):
    input_dir = ROOT_DIR / f"data/processed/datasets/{store}/{step}"
    feature_names = AGG_LEVEL[12] + CALENDAR_FEATURES + LAG_FEATURES
    categorical_features = AGG_LEVEL[12] + CALENDAR_FEATURES

    train_parquet = input_dir / "train.parquet"
    train_csv = input_dir / "train.csv"
    train_bin = input_dir / "train.lgbm"
    train_dataset = pd.read_parquet(train_parquet)
    train_dataset.to_csv(train_csv, index=False, header=False)
    train = lgb.Dataset(
        str(train_csv),
        feature_name=feature_names,
        categorical_feature=categorical_features)
    if train_bin.exists():
        train_bin.unlink()
    train.save_binary(str(train_bin))
    # train_csv.unlink()
    # train_parquet.unlink()

    val_parquet = input_dir / "val.parquet"
    val_csv = input_dir / "val.csv"
    val_bin = input_dir / "val.lgbm"
    val_dataset = pd.read_parquet(val_parquet)
    val_dataset.to_csv(val_csv, index=False, header=False)
    val = lgb.Dataset(
        str(val_csv),
        feature_name=feature_names,
        reference=train)
    if val_bin.exists():
        val_bin.unlink()
    val.save_binary(str(val_bin))
    # val_csv.unlink()


def prepare_all_datasets():
    for store in range(0, 10):
        for step in range(7, FH + 1, 7):
            prepare_dataset(TARGET, N_LAGS, store, step)


def prepare_all_dataset_binaries():
    for store in range(0, 10):
        for step in range(7, FH + 1, 7):
            prepare_dataset_binaries(store, step)


def train_step(store, step):
    print(f"Training model for store {store} and step {step}")
    input_dir = ROOT_DIR / f"data/processed/datasets/{store}/{step}"
    output_dir = create_dir(ROOT_DIR / f"models/lgbm/{store}/{step}")
    train = lgb.Dataset(str(input_dir / "train.bin"))
    val = lgb.Dataset(str(input_dir / "val.bin"))
    model = lgb.train(LGBM_PARAMS, train, valid_sets=[val], verbose_eval=False)
    model.save_model(str(output_dir / "model.txt"))


def train_store(store):
    print(f"Training model for store {store}")
    for step in range(7, FH + 1, 7):
        train_step(store, step)


def train():
    print("Start training")
    for store in range(0, 10):
        train_store(store)


def predict_step(store, step):
    print(f"Making predictions for level {store} and step {step}")
    input_dir = ROOT_DIR / f"data/processed/datasets/{store}/{step}"
    output_dir = create_dir(ROOT_DIR / f"fcst/lgbm/{store}/{step}")
    val = pd.read_parquet(input_dir / "val.parquet")
    model = lgb.Booster(model_file=str(ROOT_DIR / f"models/lgbm/{store}/{step}/model.txt"))
    fcst = val[AGG_LEVEL[12] + ["sales"]].copy()
    fcst["fcst"] = model.predict(val.drop(columns=["sales"]))
    fcst.to_parquet(output_dir / "fcst.parquet")


def predict_store(store):
    print(f"Making predictions for level {store}")
    for step in range(7, FH + 1, 7):
        predict_step(store, step)


def predict():
    print("Start predicting")
    for store in range(0, 10):
        predict_store(store)


def compile_fcst():
    for store in range(0, 10):
        print(f"Compiling forecast store {store}")
        fcst_list = []
        for step in range(7, FH + 1, 7):
            fcst_step = pd.read_parquet(ROOT_DIR / f"fcst/lgbm/{store}/{step}/fcst.parquet")
            start = fcst_step.d.min()
            if store == 1:
                step_value = fcst_step.loc[fcst_step.d == start + step - 1, :]
            else:
                step_value = fcst_step.groupby(AGG_LEVEL[12][:-1], group_keys=False).apply(
                    lambda df: df.loc[df.d == start + step - 1, :])
            fcst_list.append(step_value)
        fcst_store = pd.concat(fcst_list, axis=0).sort_values(by=AGG_LEVEL[12])
        output_dir = create_dir(ROOT_DIR / f"fcst/lgbm/{store}/fcst.parquet")
        fcst_store.to_parquet(output_dir / "fcst.parquet")
