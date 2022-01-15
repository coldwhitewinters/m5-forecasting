import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

import pickle
import itertools
from functools import partial
from joblib import Parallel, delayed

from m5.definitions import ROOT_DIR, AGG_LEVEL, N_STORES
from m5.utils import create_dir


class Naive:
    def __init__(self, **kwargs):
        self.model = None

    def train(self, y, X=None, **kwargs):
        self.model = y[-1]
        return self

    def predict(self, fh, X=None, **kwargs):
        pred = np.array(list(itertools.repeat(self.model, fh)))
        return pred


class ETS:
    def __init__(self, auto=True, **kwargs):
        self.auto = auto
        self.model_partial = partial(ETSModel, **kwargs)
        self.model = None

    def train(self, y, X=None, **kwargs):
        if self.auto:
            self.model = self.model_selection(y, **kwargs)
        else:
            self.model = self.model_partial(y).fit(**kwargs)
        return self

    def predict(self, fh, X=None, **kwargs):
        pred = self.model.forecast(fh)
        return pred

    @staticmethod
    def model_selection(y, **kwargs):
        if y.min() > 0:
            search_space = {
                "error": ["add", "mul"],
                "trend": [None, "add", "mul"],
                "damped_trend": [False, True],
                "seasonal": [None, "add", "mul"],
                "seasonal_periods": [7]}
        else:
            search_space = {
                "error": ["add"],
                "trend": [None, "add"],
                "damped_trend": [False, True],
                "seasonal": [None, "add"],
                "seasonal_periods": [7]}
        best_aicc = 1e16
        best_params = ("add", None, False, None, 1)
        for params in itertools.product(*search_space.values()):
            try:
                model = ETSModel(y, *params)
                model_fit = model.fit(**kwargs)
                if model_fit.aicc < best_aicc:
                    best_aicc = model_fit.aicc
                    best_params = params
            except ValueError:
                continue
        best_model = ETSModel(y, *best_params).fit(**kwargs)
        return best_model


class ARIMA:
    def __init__(self, **kwargs):
        self.model = pm.AutoARIMA(**kwargs)

    def train(self, y, X=None, **kwargs):
        self.model.fit(y, X, **kwargs)
        return self

    def predict(self, fh, X=None, **kwargs):
        pred = self.model.predict(n_periods=fh, X=X, **kwargs)
        return pred


class BottomUp:
    def __init__(self, model_name, model_cls, model_params=None, regressors=None, n_jobs=None):
        self.model_name = model_name
        self.model_cls = model_cls
        if model_params is None:
            self.model_params = dict()
        else:
            self.model_params = model_params
        self.regressors = regressors
        self.n_jobs = None

    def train(self, **kwargs):
        output_dir = create_dir(ROOT_DIR / f"models/{self.model_name}")
        Parallel(n_jobs=self.n_jobs)(delayed(self.train_store)(store, **kwargs) for store in range(N_STORES))
        print("Saving model...")
        file = output_dir / "model.pkl"
        with open(file, "wb") as f:
            pickle.dump(self.models_d, f)
        print("Done.")

    def train_store(self, store, **kwargs):
        self.models_d = dict()
        train_data = pd.read_parquet(ROOT_DIR / f"data/processed/stores/{store}/train.parquet")
        for item in train_data.item_id.unique():
            print(f"Training model for item {item} and store {store}   ", end="\r")
            train_item = train_data.loc[train_data.item_id == item, :]
            y = train_item.loc[:, "sales"].astype("float64").to_numpy()
            X = None
            if self.regressors is not None:
                X = train_item.loc[:, self.regressors].astype("float64").to_numpy()
            model = self.model_cls(**self.model_params)
            self.models_d[(item, store)] = model.train(y, X, **kwargs)
        print("\n")

    def predict(self, fh, **kwargs):
        output_dir = create_dir(ROOT_DIR / f"fcst/{self.model_name}/12")
        model_file = open(ROOT_DIR / f"models/{self.model_name}/model.pkl", "rb")
        self.models_d = pickle.load(model_file)
        model_file.close()
        self.fcst_l = []
        Parallel(n_jobs=self.n_jobs)(delayed(self.predict_store)(store, **kwargs) for store in range(N_STORES))
        print("Saving prediction...")
        fcst_df = pd.concat(self.fcst_l)
        fcst_df.to_parquet(output_dir / "fcst.parquet")
        self.bottom_up()
        print("Done.")

    def predict_store(self, store, fh, **kwargs):
        val_data = pd.read_parquet(ROOT_DIR / f"data/processed/stores/{store}/val.parquet")
        for item in val_data.item_id.unique():
            print(f"Making predictions for item {item} and store {store}   ", end="\r")
            val_item = val_data.loc[val_data.item_id == item, :]
            X = None
            if self.regressors is not None:
                X = val_item.loc[:, self.regressors].astype("float64").to_numpy()
            fcst = val_item.loc[:, AGG_LEVEL[12] + ["sales"]].copy()
            fcst["fcst"] = self.models_d[(item, store)].predict(fh, X, **kwargs)
            self.fcst_l.append(fcst)
        print("\n")

    def bottom_up(self):
        id_cols = pd.read_parquet(ROOT_DIR / "data/processed/id-cols.parquet")
        base_fcst = pd.read_parquet(ROOT_DIR / f"fcst/{self.model_name}/12/fcst.parquet")
        base_fcst = base_fcst.drop(columns=["item_id", "store_id"])
        base_fcst = id_cols.join(base_fcst, how="right")
        for level in range(1, 12):
            print(f"Making predictions for level {level}   ", end="\r")
            output_dir = create_dir(ROOT_DIR / f"fcst/{self.model_name}/{level}")
            fcst = base_fcst.groupby(AGG_LEVEL[level])[["sales", "fcst"]].sum().reset_index()
            fcst.to_parquet(output_dir / "fcst.parquet")
        print("\nDone")
