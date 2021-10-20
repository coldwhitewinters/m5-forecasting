import pandas as pd
import matplotlib.pyplot as plt
from m5.definitions import AGG_LEVEL


def plot_fcst(data_dir, fcst_dir, level, step, key=None, plot_tail=True):
    agg_level = AGG_LEVEL[level][:-1]
    
    fcst_file = fcst_dir / f"{level}/{step}/fcst.parquet"
    fcst = pd.read_parquet(fcst_file)
    
    train_file = data_dir / f"processed/datasets/{level}/{step}/train.parquet"
    train = pd.read_parquet(train_file)

    if key is None:
        key = tuple(0 for _ in range(len(agg_level)))

    if level == 1:
        filter_fcst = [True for _ in range(len(fcst))]
        filter_train = [True for _ in range(len(train))]
    else:
        filter_fcst = (fcst[agg_level] == key).all(axis=1)
        filter_train = (train[agg_level] == key).all(axis=1)

    f, ax = plt.subplots()
    fcst.loc[filter_fcst, ["sales", "fcst", "d"]].set_index("d").plot(ax=ax)
    train.loc[filter_train, ["sales", "d"]].set_index("d").plot(ax=ax)
    if plot_tail:
        plt.xlim(left=fcst.d.min() - 100, right=fcst.d.max() + 1)
