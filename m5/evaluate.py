import numpy as np
import pandas as pd
from m5.definitions import ROOT_DIR, FH, AGG_LEVEL
from m5.utils import create_dir


def level_accuracy(model, level):
    agg_level = AGG_LEVEL[level][:-1]
    output_dir = create_dir(ROOT_DIR / f"metrics/{model}/{level}")

    data = pd.read_parquet(ROOT_DIR / f"data/processed/levels/{level}/data.parquet")
    fcst = pd.read_parquet(ROOT_DIR / f"fcst/{model}/{level}/fcst.parquet").astype("int64")  # Avoid overflow
    data = data.loc[data.d <= data.d.max() - FH, agg_level + ["d", "sales", "dollar_sales"]]

    if level == 1:
        accuracy_df = pd.DataFrame(index=[0])
        accuracy_df["mse_naive_insample"] = data["sales"].agg(lambda x: (x.diff()**2).mean())
        accuracy_df["mse_fcst"] = ((fcst["sales"] - fcst["fcst"])**2).mean()
        accuracy_df["weights"] = 1
        accuracy_df["msse"] = accuracy_df["mse_fcst"] / accuracy_df["mse_naive_insample"]
        accuracy_df["rmsse"] = np.sqrt(accuracy_df["msse"])
        accuracy_df["wrmsse"] = accuracy_df["rmsse"] * accuracy_df["weights"]
        accuracy_df.to_csv(output_dir / "accuracy.csv", index=False)
        return accuracy_df

    total_dollar_sales = data.loc[data.d > data.d.max() - FH, "dollar_sales"].sum()
    weights = data.loc[data.d > data.d.max() - FH, :].groupby(agg_level)["dollar_sales"].agg(
        lambda x: x.sum() / total_dollar_sales).reset_index()
    weights = weights.rename(columns={"dollar_sales": "weights"})

    mse_naive_insample = data.groupby(agg_level)["sales"].agg(lambda x: (x.diff()**2).mean()).reset_index()
    mse_naive_insample = mse_naive_insample.rename(columns={"sales": "mse_naive_insample"})

    mse_fcst = fcst.groupby(agg_level).apply(lambda df: ((df["sales"] - df["fcst"])**2).mean()).reset_index()
    mse_fcst = mse_fcst.rename(columns={0: "mse_fcst"})

    accuracy_df = pd.merge(mse_fcst, mse_naive_insample, on=agg_level)
    accuracy_df = accuracy_df.merge(weights, on=agg_level)
    accuracy_df["msse"] = accuracy_df["mse_fcst"] / accuracy_df["mse_naive_insample"]
    accuracy_df["rmsse"] = np.sqrt(accuracy_df["msse"])
    accuracy_df["wrmsse"] = accuracy_df["rmsse"] * accuracy_df["weights"]

    accuracy_df.to_csv(output_dir / "accuracy.csv", index=False)
    return accuracy_df


def accuracy(model):
    acc_d = {}
    for level in range(1, 12 + 1):
        print(f"Calculating accuracy for level {level}   ", end="\r")
        level_acc = level_accuracy(model, level)
        wrmsse = level_acc["wrmsse"].sum()
        acc_d[level] = wrmsse
    acc = pd.DataFrame(acc_d, index=["wmrsse"])
    acc["Average"] = acc.T.mean()
    acc.to_csv(ROOT_DIR / f"metrics/{model}/accuracy.csv", index=False)
    return acc


def accuracy_all_models():
    pass
