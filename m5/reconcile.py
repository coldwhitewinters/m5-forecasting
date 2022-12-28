import pandas as pd
from m5.definitions import ROOT_DIR, AGG_LEVEL
from m5.utils import create_dir


def bottom_up(model_name):
    print("Making bottom up predictions...")
    id_cols = pd.read_parquet(ROOT_DIR / "data/processed/id-cols.parquet")
    base_fcst = pd.read_parquet(ROOT_DIR / f"fcst/{model_name}/12/fcst.parquet")
    base_fcst = base_fcst.drop(columns=["item_id", "store_id"])
    base_fcst = id_cols.join(base_fcst, how="right")
    for level in range(1, 12):
        # print(f"Making bottom up prediction for level {level}")
        output_dir = create_dir(ROOT_DIR / f"fcst/{model_name}/{level}")
        fcst = base_fcst.groupby(AGG_LEVEL[level])[["sales", "fcst"]].sum().reset_index()
        fcst.to_parquet(output_dir / "fcst.parquet")
