import pandas as pd
import lightgbm as lgb
from m5.definitions import AGG_LEVEL


def train_step(data_dir, model_dir, level, step, params):
    print(f"Training model for level {level} and step {step}")
    input_dir = data_dir / f"processed/datasets/{level}/{step}"
    output_dir = model_dir / f"{level}/{step}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    train = lgb.Dataset(str(input_dir / "train.bin"))
    val = lgb.Dataset(str(input_dir / "val.bin"))
    model = lgb.train(params, train, valid_sets=[val], verbose_eval=False)
    model.save_model(str(output_dir / "model.txt"))


def train_level(data_dir, model_dir, fh, level, params):
    print(f"Training model for level {level}")
    for step in range(1, fh + 1):
        train_step(data_dir, model_dir, level, step, params)


def train(data_dir, model_dir, fh, params):
    print("Start training")
    for level in range(1, 12 + 1):
        train_level(data_dir, model_dir, fh, level, params)


def predict_step(data_dir, model_dir, fcst_dir, level, step):
    print(f"Making predictions for level {level} and step {step}")
    input_dir = data_dir / f"processed/datasets/{level}/{step}"
    output_dir = fcst_dir / f"{level}/{step}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    val = pd.read_parquet(input_dir / "val.parquet")
    model = lgb.Booster(model_file=str(model_dir / f"{level}/{step}/model.txt"))
    fcst = val[AGG_LEVEL[level] + ["sales"]].copy()
    fcst["fcst"] = model.predict(val.drop(columns=["sales"]))
    fcst.to_parquet(output_dir / "fcst.parquet")


def predict_level(data_dir, model_dir, fcst_dir, fh, level):
    print(f"Making predictions for level {level}")
    for step in range(1, fh + 1):
        predict_step(data_dir, model_dir, fcst_dir, level, step)


def predict(data_dir, model_dir, fcst_dir, fh):
    print("Start predicting")
    for level in range(1, 12 + 1):
        predict_level(data_dir, model_dir, fcst_dir, fh, level)


def compile_fcst(fcst_dir, fh):
    for level in range(1, 12 + 1):
        print(f"Compiling forecast level {level}")
        fcst_list = []
        for step in range(1, fh + 1):
            print(step, end="\r")
            fcst_step = pd.read_parquet(fcst_dir / f"{level}/{step}/fcst.parquet")
            start = fcst_step.d.min()
            if level == 1:
                step_value = fcst_step.loc[fcst_step.d == start + step - 1, :]
            else:
                step_value = fcst_step.groupby(AGG_LEVEL[level][:-1], group_keys=False).apply(
                    lambda df: df.loc[df.d == start + step, :])
            fcst_list.append(step_value)
        fcst_level = pd.concat(fcst_list, axis=0).sort_values(by=AGG_LEVEL[level])
        output_dir = fcst_dir / f"{level}/final"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        fcst_level.to_parquet(output_dir / "fcst.parquet")
