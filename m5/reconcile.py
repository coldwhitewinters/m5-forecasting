import pandas as pd
from m5.definitions import AGG_LEVEL
from m5.utils import create_dir


def bottom_up(data_dir, fcst_dir, model):
    id_cols = pd.read_parquet(data_dir / "processed/id-cols.parquet")
    base_fcst = pd.read_parquet(fcst_dir / f"{model}/12/fcst.parquet")
    base_fcst = base_fcst.drop(columns=["item_id", "store_id"])
    base_fcst = id_cols.join(base_fcst, how="right")
    for level in range(1, 12):
        print(f"Making predictions for level {level}   ", end="\r")
        output_dir = create_dir(fcst_dir / f"{model}/{level}")
        fcst = base_fcst.groupby(AGG_LEVEL[level])[["sales", "fcst"]].sum().reset_index()
        fcst.to_parquet(output_dir / "fcst.parquet")
    print("\nDone.")
