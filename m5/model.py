import pandas as pd
import lightgbm as lgb
from m5.definitions import AGG_LEVEL


def train(data_dir, model_dir, level, fh, params):
    print("Start training")
    for step in range(1, 1 + fh):
        print(f"Training model for level {level} and step {step}")
        input_dir = data_dir / f"processed/datasets/{level}/{step}"
        output_dir = model_dir / f"{level}/{step}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        train = lgb.Dataset(str(input_dir / "train.bin"))
        val = lgb.Dataset(str(input_dir / "val.bin"))
        model = lgb.train(params, train, valid_sets=[val])
        model.save_model(str(output_dir / "model.txt"))
    print("Done.")


def predict(data_dir, model_dir, fcst_dir, level, fh):
    print("Start predicting")
    for step in range(1, 1 + fh):
        print(f"Predicting with model for level {level} and step {step}")
        input_dir = data_dir / f"processed/datasets/{level}/{step}"
        output_dir = fcst_dir / f"{level}/{step}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        val = pd.read_parquet(input_dir / "val.parquet")
        model = lgb.Booster(model_file=str(model_dir / f"{level}/{step}/model.txt"))
        fcst = val[AGG_LEVEL[level] + ["sales"]].copy()
        fcst["fcst"] = model.predict(val.drop(columns=["sales"]))
        fcst.to_parquet(output_dir / "fcst.parquet")
    print("Done.")
