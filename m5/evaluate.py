import numpy as np
import pandas as pd
from m5.definitions import AGG_LEVEL


def accuracy(data_dir, fcst_dir, metrics_dir, level):
    agg_level = AGG_LEVEL[level][:-1]

    data = pd.read_parquet(data_dir / f"processed/levels/level-{level}.parquet")
    fcst = pd.read_parquet(fcst_dir / f"fcst-{level}.parquet")
    data = data.loc[data.d < 1886, agg_level + ["d", "sales", "dollar_sales"]]  # Replace hard coded number

    if level == 1:
        accuracy_df = pd.DataFrame(index=[0])
        accuracy_df["mse_naive_insample"] = data["sales"].agg(lambda x: (x.diff()**2).mean())
        accuracy_df["mse_fcst"] = ((fcst["sales"] - fcst["fcst"])**2).mean()
        accuracy_df["weights"] = 1
        accuracy_df["msse"] = accuracy_df["mse_fcst"] / accuracy_df["mse_naive_insample"]
        accuracy_df["rmsse"] = np.sqrt(accuracy_df["msse"])
        accuracy_df["wrmsse"] = accuracy_df["rmsse"] * accuracy_df["weights"]
        accuracy_df.to_csv(metrics_dir / f"accuracy-{level}.csv", index=False)
        return accuracy_df

    total_dollar_sales = data.loc[data.d >= 1858, "dollar_sales"].sum()  # Replace hard coded number
    weights = data.loc[data.d >= 1858, :].groupby(agg_level)["dollar_sales"].agg(
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
    accuracy_df.to_csv(metrics_dir / f"accuracy-{level}.csv", index=False)
    return accuracy_df
