# %% Setup

import sys
sys.path.append(".")

import papermill as pm
import pandas as pd
import m5.config as cfg


def preprocessing():
    pm.execute_notebook(
        cfg.NOTEBOOKS_DIR / "preprocessing.ipynb",
        cfg.NOTEBOOKS_DIR / "preprocessing-output.ipynb"
    )


def forecast():
    for level in range(1, 12 + 1):
        pm.execute_notebook(
            cfg.NOTEBOOKS_DIR / "forecast.ipynb",
            cfg.NOTEBOOKS_DIR / f"forecast-{level}.ipynb",
            parameters=dict(level=level)
        )


def collect_metrics():
    acc_d = {}
    for level in range(1, 12 + 1):
        level_acc = pd.read_csv(cfg.METRICS_DIR / f"accuracy-{level}.csv")
        wrmsse = level_acc["wrmsse"].sum()
        acc_d[level] = wrmsse
    acc = pd.DataFrame(acc_d, index=["wmrsse"])
    acc["Average"] = acc.T.mean()
    acc.to_csv(cfg.METRICS_DIR / "accuracy_final.csv", index=False)
    print(acc.T)


# %% Main

if __name__ == "__main__":
    preprocessing()
    forecast()
    collect_metrics()
