import numpy as np
import pandas as pd
from m5.definitions import AGG_LEVEL


def accuracy(data_dir, fcst_dir, metrics_dir, fh, level, step="final"):
    print(f"Calculating accuracy for level {level}")

    agg_level = AGG_LEVEL[level][:-1]
    output_dir = metrics_dir / f"{level}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    data = pd.read_parquet(data_dir / f"processed/levels/{level}/data.parquet")
    fcst = pd.read_parquet(fcst_dir / f"{level}/{step}/fcst.parquet")
    data = data.loc[data.d <= data.d.max() - fh, agg_level + ["d", "sales", "dollar_sales"]]
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

    total_dollar_sales = data.loc[data.d > data.d.max() - 2 * fh, "dollar_sales"].sum()
    weights = data.loc[data.d > data.d.max() - 2 * fh, :].groupby(agg_level)["dollar_sales"].agg(
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


def accuracy_all_levels(data_dir, fcst_dir, metrics_dir, fh, step="final"):
    for level in range(1, 12 + 1):
        accuracy(data_dir, fcst_dir, metrics_dir, fh, level, step)


def collect_metrics(metrics_dir):
    acc_d = {}
    for level in range(1, 12 + 1):
        level_acc = pd.read_csv(metrics_dir / f"{level}/accuracy.csv")
        wrmsse = level_acc["wrmsse"].sum()
        acc_d[level] = wrmsse
    acc = pd.DataFrame(acc_d, index=["wmrsse"])
    acc["Average"] = acc.T.mean()
    acc.to_csv(metrics_dir / "accuracy_final.csv", index=False)
    print(acc.T)
