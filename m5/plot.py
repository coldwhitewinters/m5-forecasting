import pandas as pd
import matplotlib.pyplot as plt
from m5.definitions import AGG_LEVEL


def plot_fcst(data_dir, fcst_dir, multi_step=True, level=1, key=None, plot_tail=True):
    agg_level = AGG_LEVEL[level][:-1]

    if multi_step:
        step = "final"
    else:
        step = 28

    fcst_file = fcst_dir / f"{level}/{step}/fcst.parquet"
    fcst = pd.read_parquet(fcst_file)

    data_file = data_dir / f"processed/levels/{level}/data.parquet"
    data = pd.read_parquet(data_file)

    if key is None:
        key = tuple(0 for _ in range(len(agg_level)))

    if level == 1:
        filter_fcst = [True for _ in range(len(fcst))]
        filter_data = [True for _ in range(len(data))]
    else:
        filter_fcst = (fcst[agg_level] == key).all(axis=1)
        filter_data = (data[agg_level] == key).all(axis=1)

    f, ax = plt.subplots()
    data.loc[filter_data, ["sales", "d"]].set_index("d").plot(ax=ax)
    fcst.loc[filter_fcst, ["fcst", "d"]].set_index("d").plot(ax=ax)
    if plot_tail:
        plt.xlim(left=fcst.d.min() - 100, right=fcst.d.max() + 1)
