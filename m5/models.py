import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import lightgbm as lgb

import pickle
import itertools
from functools import partial
from joblib import Parallel, delayed

from m5.reconcile import bottom_up
from m5.utils import create_dir
from m5.definitions import (
    ROOT_DIR, AGG_LEVEL, N_STORES, STEP_RANGE)


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
    def __init__(
        self,
        model_name,
        model_cls,
        model_params=None,
        regressors=None,
        n_jobs=None,
        parallel_backend="loky",
    ):
        self.model_name = model_name
        self.model_cls = model_cls
        if model_params is None:
            self.model_params = dict()
        else:
            self.model_params = model_params
        self.regressors = regressors
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

    def train(self, **kwargs):
        print("Start training...")
        output_dir = create_dir(ROOT_DIR / f"models/{self.model_name}")
        model_l = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(
            delayed(self.train_store)(store, **kwargs) for store in range(N_STORES))
        print("Saving models...")
        file = output_dir / "model.pkl"
        with open(file, "wb") as f:
            pickle.dump(model_l, f)
        print("Done.")

    def train_store(self, store, **kwargs):
        print(f"Training models for store {store}")
        train_data = pd.read_parquet(ROOT_DIR / f"data/processed/stores/{store}/train.parquet")
        model_l = []
        for item in train_data.item_id.unique():
            # print(f"Training model for store {store} and item {item}")
            train_item = train_data.loc[train_data.item_id == item, :]
            y = train_item.loc[:, "sales"].astype("float64").to_numpy()
            X = None
            if self.regressors is not None:
                X = train_item.loc[:, self.regressors].astype("float64").to_numpy()
            model = self.model_cls(**self.model_params).train(y, X, **kwargs)
            model_l.append(model)
        return model_l

    def predict(self, fh, **kwargs):
        print("Start predicting...")
        output_dir = create_dir(ROOT_DIR / f"fcst/{self.model_name}/12")
        fcst_l = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(
            delayed(self.predict_store)(store, fh, **kwargs) for store in range(N_STORES))
        print("Compiling predictions...")
        fcst_df = pd.concat(fcst_l)
        fcst_df.to_parquet(output_dir / "fcst.parquet")
        bottom_up(self.model_name)
        print("Done.")

    def predict_store(self, store, fh, **kwargs):
        print(f"Making predictions for store {store}")
        val_data = pd.read_parquet(ROOT_DIR / f"data/processed/stores/{store}/val.parquet")
        with open(ROOT_DIR / f"models/{self.model_name}/model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        fcst_l = []
        for item in val_data.item_id.unique():
            # print(f"Making predictions for store {store} and item {item}")
            val_item = val_data.loc[val_data.item_id == item, :]
            X = None
            if self.regressors is not None:
                X = val_item.loc[:, self.regressors].astype("float64").to_numpy()
            fcst = val_item.loc[:, AGG_LEVEL[12] + ["sales"]].copy()
            fcst["fcst"] = model[store][item].predict(fh, X, **kwargs)
            fcst_l.append(fcst)
        fcst_df = pd.concat(fcst_l)
        return fcst_df


class LGBM:
    def __init__(self, model_name, model_params):
        self.model_name = model_name
        self.model_params = model_params

    def train_step(self, store, step):
        print(f"Training model for store {store} and step {step}")
        input_dir = ROOT_DIR / f"data/processed/datasets/{store}/{step}"
        output_dir = create_dir(ROOT_DIR / f"models/{self.model_name}/{store}/{step}")
        train = lgb.Dataset(str(input_dir / "train.lgbm"))
        val = lgb.Dataset(str(input_dir / "val.lgbm"))
        model = lgb.train(self.model_params, train, valid_sets=[val], verbose_eval=False)
        model.save_model(str(output_dir / "model.txt"))

    def train_store(self, store):
        print(f"Training model for store {store}")
        for step in STEP_RANGE:
            self.train_step(store, step)

    def train(self):
        print("Start training...")
        for store in range(N_STORES):
            self.train_store(store)
        print("Done.")

    def predict_step(self, store, step):
        print(f"Making predictions for level {store} and step {step}")
        input_dir = ROOT_DIR / f"data/processed/datasets/{store}/{step}"
        output_dir = create_dir(ROOT_DIR / f"fcst/{self.model_name}/12/{store}/{step}")
        val = pd.read_parquet(input_dir / "val.parquet")
        model = lgb.Booster(model_file=str(ROOT_DIR / f"models/{self.model_name}/{store}/{step}/model.txt"))
        fcst = val[AGG_LEVEL[12] + ["sales"]].copy()
        fcst["fcst"] = model.predict(val.drop(columns=["sales"]))
        fcst.to_parquet(output_dir / "fcst.parquet")

    def predict_store(self, store):
        print(f"Making predictions for level {store}")
        for step in STEP_RANGE:
            self.predict_step(store, step)

    def predict(self):
        print("Start predicting...")
        for store in range(N_STORES):
            self.predict_store(store)
        print("Compiling predictions...")
        self.compile_fcst()
        bottom_up(self.model_name)
        print("Done.")

    def compile_fcst(self):
        output_dir = create_dir(ROOT_DIR / f"fcst/{self.model_name}/12")
        fcst_l = []
        for store in range(N_STORES):
            print(f"Compiling forecast store {store}")
            for step in STEP_RANGE:
                fcst_step = pd.read_parquet(ROOT_DIR / f"fcst/{self.model_name}/12/{store}/{step}/fcst.parquet")
                start = fcst_step.d.min()
                interval = range(start + step - 7, start + step)
                if store == 1:
                    step_value = fcst_step.loc[fcst_step.d.isin(interval), :]
                else:
                    step_value = fcst_step.groupby(AGG_LEVEL[12][:-1], group_keys=False).apply(
                        lambda df: df.loc[df.d.isin(interval), :])
                fcst_l.append(step_value)
        fcst_df = pd.concat(fcst_l, axis=0).sort_values(by=AGG_LEVEL[12])
        fcst_df.to_parquet(output_dir / "fcst.parquet")
