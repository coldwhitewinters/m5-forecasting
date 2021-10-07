import numpy as np
import pandas as pd


def rmsse(y_train, y_val, y_fcst):
    sqerr = (y_val - y_fcst)**2
    sqdiffs = y_train.diff()**2
    rmsse_value = np.sqrt(sqerr.mean() / sqdiffs.mean())
    return rmsse_value


def calculate_rmsse(train, val, fcst):
    mse_naive_insample = train.groupby("id")["sales"].agg(lambda x: (x.diff()**2).mean())
    mse_fcst = pd.merge(fcst, val, on=["id", "d"]).set_index("d").groupby("id").apply(
        lambda df: ((df["sales"] - df["fcst"])**2).mean())
    msse = mse_fcst / mse_naive_insample
    msse = msse[(msse != np.inf) & (~msse.isna())]
    rmsse = np.sqrt(msse).sort_values()
    return rmsse
