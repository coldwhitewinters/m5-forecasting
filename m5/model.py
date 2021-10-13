import pandas as pd
import lightgbm as lgb
from m5.definitions import AGG_LEVEL


def train(data_dir, model_dir, level, params):
    print("Training model...")
    train = lgb.Dataset(str(data_dir / f"processed/datasets/{level}/train.bin"))
    val = lgb.Dataset(str(data_dir / f"processed/datasets/{level}/val.bin"))
    model = lgb.train(params, train, valid_sets=[val])
    model.save_model(str(model_dir / f"model-{level}.txt"))
    print("Done.")


def predict(data_dir, model_dir, fcst_dir, level):
    val = pd.read_parquet(data_dir / f"processed/datasets/{level}/val.parquet")
    model = lgb.Booster(model_file=str(model_dir / f"model-{level}.txt"))
    fcst = val[AGG_LEVEL[level] + ["sales"]].copy()
    fcst["fcst"] = model.predict(val.drop(columns=["sales"]))
    fcst.to_parquet(fcst_dir / f"fcst-{level}.parquet")
    return fcst
