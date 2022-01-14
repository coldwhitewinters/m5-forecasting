import zipfile
import pandas as pd
from m5.features import build_lag_features
from m5.utils import get_columns, move_column, create_dir
from m5.config import ROOT_DIR
from m5.definitions import AGG_LEVEL, ID_COLS, CALENDAR_FEATURES


def unzip_data():
    with zipfile.ZipFile(ROOT_DIR / "data/m5-forecasting-accuracy.zip") as zip:
        zip.extractall(ROOT_DIR / "data")


def load_data(task="train"):
    if task == "train":
        sales = pd.read_csv(ROOT_DIR / "data/sales_train_validation.csv")
    elif task == "test":
        sales = pd.read_csv(ROOT_DIR / "data/sales_train_evaluation.csv")
    calendar = pd.read_csv(ROOT_DIR / "data/calendar.csv")
    prices = pd.read_csv(ROOT_DIR / "data/sell_prices.csv")
    return sales, calendar, prices


def convert_dtypes(sales, calendar, prices):
    id_cols = sales.columns[:6]
    sales.loc[:, "id"] = sales.loc[:, "id"].str[:-11]
    sales.loc[:, id_cols] = sales.loc[:, id_cols].astype("category")

    calendar.loc[:, "date"] = pd.to_datetime(calendar.loc[:, "date"])
    calendar.loc[:, "wm_yr_wk"] = calendar.loc[:, "wm_yr_wk"].astype("int16")
    calendar.loc[:, "weekday"] = calendar.loc[:, "weekday"].astype("category")
    calendar.loc[:, "wday"] = calendar.loc[:, "wday"].astype("int8")
    calendar.loc[:, "month"] = calendar.loc[:, "month"].astype("int8")
    calendar.loc[:, "year"] = calendar.loc[:, "year"].astype("int16")
    calendar.loc[:, "event_name_1"] = calendar.loc[:, "event_name_1"].fillna("NoEvent").astype("category")
    calendar.loc[:, "event_type_1"] = calendar.loc[:, "event_type_1"].fillna("NoEvent").astype("category")
    calendar.loc[:, "event_name_2"] = calendar.loc[:, "event_name_2"].fillna("NoEvent").astype("category")
    calendar.loc[:, "event_type_2"] = calendar.loc[:, "event_type_2"].fillna("NoEvent").astype("category")
    calendar.loc[:, "snap_CA"] = calendar.loc[:, "snap_CA"].astype("bool")
    calendar.loc[:, "snap_TX"] = calendar.loc[:, "snap_TX"].astype("bool")
    calendar.loc[:, "snap_WI"] = calendar.loc[:, "snap_WI"].astype("bool")

    prices.loc[:, ["store_id", "item_id"]] = prices.loc[:, ["store_id", "item_id"]].astype("category")
    prices.loc[:, "wm_yr_wk"] = prices.loc[:, "wm_yr_wk"].astype("int16")
    prices.loc[:, "sell_price"] = prices.loc[:, "sell_price"].astype("float32")


def pivot_longer_sales(sales):
    data = pd.melt(sales, id_vars=sales.columns[:6], var_name="d", value_name="sales")
    data.loc[:, "sales"] = data.loc[:, "sales"].astype("int16")
    return data


def remove_leading_zero_sales(data):
    cu_sales = data.groupby(["id"])["sales"].cumsum()
    data = data[cu_sales > 0]
    return data


def merge_calendar(data, calendar):
    data = data.merge(calendar.drop(columns=["weekday"]), on="d", how="left")
    return data


def merge_prices(data, prices):
    data = data.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    return data


def add_dollar_sales(data):
    data["dollar_sales"] = data["sales"] * data["sell_price"]
    return data


def base_data_cleanup(data):
    data["d"] = data["d"].str[2:].astype("int16")
    data = data.drop(columns=["wm_yr_wk"])
    dt = data["date"]
    data = data.drop(columns=["date"])
    data.insert(7, "date", dt)
    data = data.sort_values(by=["id", "d"])
    return data


def prepare_base_data(task="train"):
    output_dir = create_dir(ROOT_DIR / "data/processed")
    sales, calendar, prices = load_data(task)
    convert_dtypes(sales, calendar, prices)
    data = sales  # Alias
    data = pivot_longer_sales(data)
    data = remove_leading_zero_sales(data)
    data = merge_calendar(data, calendar)
    data = merge_prices(data, prices)
    data = add_dollar_sales(data)
    data = base_data_cleanup(data)
    data.to_parquet(output_dir / "base.parquet")


def category_to_int(data):
    data["id"] = pd.factorize(data["id"])[0].astype("int16")
    data["item_id"] = pd.factorize(data["item_id"])[0].astype("int16")
    data["dept_id"] = pd.factorize(data["dept_id"])[0].astype("int8")
    data["cat_id"] = pd.factorize(data["cat_id"])[0].astype("int8")
    data["store_id"] = pd.factorize(data["store_id"])[0].astype("int8")
    data["state_id"] = pd.factorize(data["state_id"])[0].astype("int8")
    data["event_name_1"] = pd.factorize(data["event_name_1"])[0].astype("int8")
    data["event_type_1"] = pd.factorize(data["event_type_1"])[0].astype("int8")
    data["event_name_2"] = pd.factorize(data["event_name_2"])[0].astype("int8")
    data["event_type_2"] = pd.factorize(data["event_type_2"])[0].astype("int8")
    data["snap_CA"] = data["snap_CA"].astype("int8")
    data["snap_TX"] = data["snap_TX"].astype("int8")
    data["snap_WI"] = data["snap_WI"].astype("int8")


def agg_data(data, lvl):
    data_agg = data.groupby(AGG_LEVEL[lvl]).agg({
        "sales": "sum",
        "dollar_sales": "sum",
    }).reset_index()

    data_agg["d"] = data_agg["d"].astype("int16")
    data_agg["sales"] = data_agg["sales"].astype("int32")
    if "id" in data_agg.columns:
        data_agg["id"] = data_agg["id"].astype("int16")
    if "item_id" in data_agg.columns:
        data_agg["item_id"] = data_agg["item_id"].astype("int16")
    if "dept_id" in data_agg.columns:
        data_agg["dept_id"] = data_agg["dept_id"].astype("int8")
    if "cat_id" in data_agg.columns:
        data_agg["cat_id"] = data_agg["cat_id"].astype("int8")
    if "store_id" in data_agg.columns:
        data_agg["store_id"] = data_agg["store_id"].astype("int8")
    if "state_id" in data_agg.columns:
        data_agg["state_id"] = data_agg["state_id"].astype("int8")
    return data_agg


def prepare_agg_levels():
    input_file = ROOT_DIR / "data/processed/base.parquet"
    level_12_cols = [
        'item_id', 'store_id', 'd',
        'sales', 'dollar_sales', 'wday', 'month', 'year',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI']

    print("Preparing agg level 12 ", end="\r")
    base_data = pd.read_parquet(input_file)
    category_to_int(base_data)
    base_data = base_data.drop(columns=["date", "sell_price"])
    base_data = base_data.reset_index(drop=True)
    base_data.to_parquet(ROOT_DIR / "data/processed/base-numeric.parquet")
    base_data[ID_COLS].to_parquet(ROOT_DIR / "data/processed/id-cols.parquet")
    calendar = base_data[["d"] + CALENDAR_FEATURES].drop_duplicates()
    output_dir = create_dir(ROOT_DIR / "data/processed/levels/12")
    base_data[level_12_cols].to_parquet(output_dir / "data.parquet")

    for lvl in range(11, 0, -1):
        print(f"Preparing agg level {lvl} ", end="\r")
        df_agg = agg_data(base_data, lvl)
        df_agg = df_agg.merge(calendar, on=["d"])
        output_dir = create_dir(ROOT_DIR / f"data/processed/levels/{lvl}")
        df_agg.to_parquet(output_dir / "data.parquet")
    print("\nDone.")


def prepare_store_data():
    data = pd.read_parquet(ROOT_DIR / "data/processed/levels/12/data.parquet")
    for store in data.store_id.unique():
        print(f"Preparing store data {store}", end="\r")
        output_dir = create_dir(ROOT_DIR / f"data/processed/stores/{store}")
        store_data = data[data.store_id == store]
        train_data = store_data.groupby(["item_id", "store_id"], group_keys=False).apply(lambda df: df.iloc[:-28])
        val_data = store_data.groupby(["item_id", "store_id"], group_keys=False).apply(lambda df: df.iloc[-28:])
        store_data.to_parquet(output_dir / "data.parquet")
        train_data.to_parquet(output_dir / "train.parquet")
        val_data.to_parquet(output_dir / "val.parquet")
    print("\nDone.")


def build_lags(data, target, step, lags):
    id_cols = get_columns(data, lambda x: x.endswith("id"))
    if not id_cols:
        dataset = build_lag_features(data, target, step, lags)
    else:
        dataset = data.groupby(id_cols, group_keys=False).apply(
            lambda df: build_lag_features(df, target, step, lags))
    return dataset


def prepare_dataset(target, fh, lags, level, step):
    print(f"Preparing dataset for level {level} and step {step}")
    input_file = ROOT_DIR / f"data/processed/levels/{level}/data.parquet"
    output_dir = create_dir(ROOT_DIR / f"data/processed/datasets/{level}/{step}")
    data = pd.read_parquet(input_file)
    move_column(data, target)
    data.drop(columns=["dollar_sales"], inplace=True)
    dataset = build_lags(data, target, step, lags)

    N = dataset.d.max()
    train = dataset.loc[dataset.d <= (N - fh), :]
    val = dataset.loc[dataset.d > (N - fh), :]

    train.to_parquet(output_dir / "train.parquet")
    val.to_parquet(output_dir / "val.parquet")
