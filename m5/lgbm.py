import pandas as pd
import lightgbm as lgb
from m5.config import ROOT_DIR
from m5.definitions import AGG_LEVEL, CALENDAR_FEATURES, LAG_FEATURES
from m5.utils import create_dir


def prepare_dataset_binaries(level, step):
    input_dir = ROOT_DIR / f"data/processed/datasets/{level}/{step}"
    feature_names = AGG_LEVEL[level] + CALENDAR_FEATURES + LAG_FEATURES
    categorical_features = AGG_LEVEL[level] + CALENDAR_FEATURES

    train_parquet = input_dir / "train.parquet"
    train_csv = input_dir / "train.csv"
    train_bin = input_dir / "train.bin"
    train_dataset = pd.read_parquet(train_parquet)
    train_dataset.to_csv(train_csv, index=False, header=False)
    train = lgb.Dataset(
        str(train_csv),
        feature_name=feature_names,
        categorical_feature=categorical_features)
    if train_bin.exists():
        train_bin.unlink()
    train.save_binary(str(train_bin))
    train_csv.unlink()
    train_parquet.unlink()

    val_parquet = input_dir / "val.parquet"
    val_csv = input_dir / "val.csv"
    val_bin = input_dir / "val.bin"
    val_dataset = pd.read_parquet(val_parquet)
    val_dataset.to_csv(val_csv, index=False, header=False)
    val = lgb.Dataset(
        str(val_csv),
        feature_name=feature_names,
        reference=train)
    if val_bin.exists():
        val_bin.unlink()
    val.save_binary(str(val_bin))
    val_csv.unlink()


def prepare_all_dataset_binaries():
    for level in range(1, 12 + 1):
        for step in STEP_RANGE:
            prepare_dataset_binaries(level, step)


def train_step(data_dir, model_dir, level, step, params):
    print(f"Training model for level {level} and step {step}")
    input_dir = data_dir / f"processed/datasets/{level}/{step}"
    output_dir = create_dir(model_dir / f"{level}/{step}")
    train = lgb.Dataset(str(input_dir / "train.bin"))
    val = lgb.Dataset(str(input_dir / "val.bin"))
    model = lgb.train(params, train, valid_sets=[val], verbose_eval=False)
    model.save_model(str(output_dir / "model.txt"))


def train_level(data_dir, model_dir, level, params):
    print(f"Training model for level {level}")
    for step in STEP_RANGE:
        train_step(data_dir, model_dir, level, step, params)


def train(data_dir, model_dir, params):
    print("Start training")
    for level in range(1, 12 + 1):
        train_level(data_dir, model_dir, level, params)


def predict_step(data_dir, model_dir, fcst_dir, level, step):
    print(f"Making predictions for level {level} and step {step}")
    input_dir = data_dir / f"processed/datasets/{level}/{step}"
    output_dir = create_dir(fcst_dir / f"{level}/{step}")
    val = pd.read_parquet(input_dir / "val.parquet")
    model = lgb.Booster(model_file=str(model_dir / f"{level}/{step}/model.txt"))
    fcst = val[AGG_LEVEL[level] + ["sales"]].copy()
    fcst["fcst"] = model.predict(val.drop(columns=["sales"]))
    fcst.to_parquet(output_dir / "fcst.parquet")


def predict_level(data_dir, model_dir, fcst_dir, level):
    print(f"Making predictions for level {level}")
    for step in STEP_RANGE:
        predict_step(data_dir, model_dir, fcst_dir, level, step)


def predict(data_dir, model_dir, fcst_dir):
    print("Start predicting")
    for level in range(1, 12 + 1):
        predict_level(data_dir, model_dir, fcst_dir, level)


def compile_fcst(fcst_dir):
    for level in range(1, 12 + 1):
        print(f"Compiling forecast level {level}")
        fcst_list = []
        for step in STEP_RANGE:
            print(step, end="\r")
            fcst_step = pd.read_parquet(fcst_dir / f"{level}/{step}/fcst.parquet")
            start = fcst_step.d.min()
            if level == 1:
                step_value = fcst_step.loc[fcst_step.d == start + step - 1, :]
            else:
                step_value = fcst_step.groupby(AGG_LEVEL[level][:-1], group_keys=False).apply(
                    lambda df: df.loc[df.d == start + step - 1, :])
            fcst_list.append(step_value)
        fcst_level = pd.concat(fcst_list, axis=0).sort_values(by=AGG_LEVEL[level])
        output_dir = create_dir(fcst_dir / f"{level}/final")
        fcst_level.to_parquet(output_dir / "fcst.parquet")
