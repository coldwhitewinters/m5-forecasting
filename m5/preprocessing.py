import zipfile
import pandas as pd
from m5.features import build_lag_features
from m5.utils import get_columns, move_column
from m5.definitions import AGG_LEVEL, CALENDAR_FEATURES, LAG_FEATURES
import lightgbm as lgb


def unzip_data(data_dir):
    with zipfile.ZipFile(data_dir / "m5-forecasting-accuracy.zip") as zip:
        zip.extractall(data_dir)


def load_data(data_dir, type="train"):
    if type == "train":
        sales = pd.read_csv(data_dir / "sales_train_validation.csv")
    elif type == "test":
        sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    prices = pd.read_csv(data_dir / "sell_prices.csv")
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


def prepare_base_data(data_dir, unzip=False):
    if unzip:
        unzip_data(data_dir)

    output_dir = data_dir / "processed"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    sales, calendar, prices = load_data(data_dir)
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


def prepare_agg_levels(data_dir):
    input_file = data_dir / "processed/base.parquet"

    level_12_cols = [
        'item_id', 'store_id', 'd',
        'sales', 'dollar_sales', 'wday', 'month', 'year',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI']

    print("Preparing agg level 12")
    base_data = pd.read_parquet(input_file)
    category_to_int(base_data)
    base_data = base_data.drop(columns=["date", "sell_price"])
    base_data = base_data.reset_index(drop=True)
    calendar = base_data[["d"] + CALENDAR_FEATURES].drop_duplicates()
    output_dir = data_dir / "processed/levels/12"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    base_data[level_12_cols].to_parquet(output_dir / "data.parquet")

    for lvl in range(1, 12):
        print(f"Preparing agg level {lvl}")
        df_agg = agg_data(base_data, lvl)
        df_agg = df_agg.merge(calendar, on=["d"])
        output_dir = data_dir / f"processed/levels/{lvl}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        df_agg.to_parquet(output_dir / "data.parquet")


def build_lags(data, target, step, lags):
    id_cols = get_columns(data, lambda x: x.endswith("id"))
    if not id_cols:
        dataset = build_lag_features(data, target, step, lags)
    else:
        dataset = data.groupby(id_cols, group_keys=False).apply(
            lambda df: build_lag_features(df, target, step, lags))
    return dataset


def prepare_dataset(data_dir, target, fh, lags, level, step):
    print(f"Preparing dataset for level {level} and step {step}")
    input_file = data_dir / f"processed/levels/{level}/data.parquet"
    output_dir = data_dir / f"processed/datasets/{level}/{step}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    data = pd.read_parquet(input_file)
    move_column(data, target)
    data.drop(columns=["dollar_sales"], inplace=True)
    dataset = build_lags(data, target, step, lags)

    N = dataset.d.max()
    train = dataset.loc[dataset.d <= (N - fh), :]
    val = dataset.loc[dataset.d > (N - fh), :]

    train.to_parquet(output_dir / "train.parquet")
    val.to_parquet(output_dir / "val.parquet")


def prepare_all_datasets(data_dir, target, fh, lags):
    for level in range(1, 12 + 1):
        for step in range(1, fh + 1):
            prepare_dataset(data_dir, target, fh, lags, level, step)


def prepare_dataset_binaries(data_dir, level, step):
    input_dir = data_dir / f"processed/datasets/{level}/{step}"
    feature_names = AGG_LEVEL[level] + CALENDAR_FEATURES + LAG_FEATURES
    categorical_features = AGG_LEVEL[level] + CALENDAR_FEATURES

    train_parquet = input_dir / "train.parquet"
    train_csv = input_dir / "train.csv"
    train_bin = input_dir / "train.bin"
    train_dataset = pd.read_parquet(train_parquet)
    train_dataset.to_csv(train_csv, index=False, header=False)
    train = lgb.Dataset(
        str(train_csv),
        feature_name=feature_names,
        categorical_feature=categorical_features)
    if train_bin.exists():
        train_bin.unlink()
    train.save_binary(str(train_bin))
    train_csv.unlink()
    train_parquet.unlink()

    val_parquet = input_dir / "val.parquet"
    val_csv = input_dir / "val.csv"
    val_bin = input_dir / "val.bin"
    val_dataset = pd.read_parquet(val_parquet)
    val_dataset.to_csv(val_csv, index=False, header=False)
    val = lgb.Dataset(
        str(val_csv),
        feature_name=feature_names,
        reference=train)
    if val_bin.exists():
        val_bin.unlink()
    val.save_binary(str(val_bin))
    val_csv.unlink()


def prepare_all_dataset_binaries(data_dir, fh):
    for level in range(1, 12 + 1):
        for step in range(1, fh + 1):
            prepare_dataset_binaries(data_dir, level, step)
