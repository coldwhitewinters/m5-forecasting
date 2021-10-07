import pandas as pd
import zipfile
from m5.features import build_lag_features
from m5.utils import get_columns
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
    sales, calendar, prices = load_data(data_dir)
    convert_dtypes(sales, calendar, prices)
    data = sales  # Alias
    data = pivot_longer_sales(data)
    data = remove_leading_zero_sales(data)
    data = merge_calendar(data, calendar)
    data = merge_prices(data, prices)
    data = base_data_cleanup(data)
    data.to_parquet(data_dir / "processed/base.parquet")


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


def move_target_to_first_col(data, target):
    values = data[target]
    data.drop(columns=[target], inplace=True)
    data.insert(0, target, values)


def drop_date(data):
    data.drop(columns=["date"], inplace=True)


def build_lags(data, target, step, lags):
    dataset = data.groupby("id", group_keys=False).apply(
        lambda df: build_lag_features(df, target, step, lags))
    return dataset


def prepare_dataset(data_dir, target, step, lags):
    data = pd.read_parquet(data_dir / "base.parquet")
    move_target_to_first_col(data, target)
    drop_date(data)
    category_to_int(data)
    dataset = build_lags(data, target, step, lags)
    dataset.to_csv(data_dir / "dataset.csv", index=False, header=False)
    dataset.to_parquet(data_dir / "dataset.parquet")


def prepare_train_val_split(data_dir, fh):
    dataset = pd.read_parquet(data_dir / "dataset.parquet")
    N = dataset.d.max()
    train = dataset[(dataset.d <= N - fh)]
    val = dataset[(dataset.d > N - fh)]
    train.to_csv(data_dir / "train.csv", index=False, header=False)
    train.to_parquet(data_dir / "train.parquet")
    val.to_csv(data_dir / "val.csv", index=False, header=False)
    val.to_parquet(data_dir / "val.parquet")


def prepare_dataset_binaries(data_dir, feature_names, categorical_features):
    train_path = str(data_dir / "train.csv")
    val_path = str(data_dir / "val.csv")
    train = lgb.Dataset(
        train_path,
        feature_name=feature_names,
        categorical_feature=categorical_features,
    )
    val = lgb.Dataset(
        val_path,
        feature_name=feature_names,
        reference=train,
    )
    train.save_binary(str(data_dir / "train.bin"))
    val.save_binary(str(data_dir / "val.bin"))


def agg_data(data, lvl, cols):
    agg_level = {
        1: ['d'],
        2: ['state_id', 'd'],
        3: ['store_id', 'd'],
        4: ['cat_id', 'd'],
        5: ['dept_id', 'd'],
        6: ['state_id', 'cat_id', 'd'],
        7: ['state_id', 'dept_id', 'd'],
        8: ['store_id', 'cat_id', 'd'],
        9: ['store_id', 'dept_id', 'd'],
        10: ['item_id', 'd'],
        11: ['item_id', 'state_id', 'd'],
        12: ['item_id', 'store_id', 'd'],
    }
    
    data_agg = data.groupby(agg_level[lvl])[cols].sum().reset_index()
    return data_agg


def prepare_agg_level(data_dir, df, lvl, cols):
    df_agg = agg_data(df, lvl, cols)
    id_cols = get_columns(df_agg, lambda x: x.endswith("id"))
    df_agg.insert(0, "id", str(lvl))
    for col in id_cols:
        df_agg["id"] = df_agg["id"] + "-" + df_agg[col].apply(str)
    df_agg = df_agg.drop(columns=id_cols)
    df_agg["id"] = df_agg["id"].astype("category")
    df_agg["d"] = df_agg["d"].astype("int16")
    df_agg.to_parquet(data_dir / "temp" / f"df_agg_level_{lvl}.parquet")


def prepare_all_agg_levels(data_dir, df, cols):
    for lvl in range(1, 12 + 1):
        prepare_agg_level(data_dir, df, lvl, cols)


def bottom_up(data_dir, df, cols):
    prepare_all_agg_levels(data_dir, df, cols)
    levels = []
    for lvl in range(1, 12 + 1):
        df_agg = pd.read_parquet(data_dir / "temp" / f"df_agg_level_{lvl}.parquet")
        levels.append(df_agg)
    df_bu = pd.concat(levels)[["id", "d"] + cols]
    return df_bu


def prepare_train_bu(data_dir):
    train = pd.read_parquet(data_dir / "train.parquet")
    train_bu = bottom_up(data_dir, train, ["sales"])
    train_bu.to_parquet(data_dir / "train_bu.parquet")
