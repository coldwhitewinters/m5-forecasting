import m5.config as cfg


def build_lag_features(df, target, fh, n_lags):
    for lag in range(0, n_lags):
        df[f"{target}_lag_{lag + 1}"] = df["sales"].shift(fh + lag).astype("float32")
    df.dropna(inplace=True)
    lag_columns = [col for col in df.columns if col.startswith("sales_lag")]
    df[lag_columns] = df[lag_columns].astype("int16")
    return df


def build_descriptive_features(df, col):
    df[f"{col}_min"] = df[col].min()
    df[f"{col}_max"] = df[col].max()
    df[f"{col}_mean"] = df[col].mean()
    df[f"{col}_std"] = df[col].std()
    return df


def build_scaled_features(df, col):
    min = df[col].min()
    max = df[col].max()
    mu = df[col].mean()
    sigma = df[col].std()
    df[f"{col}_min_max"] = df[col] / (max - min)
    df[f"{col}_standard"] = (df[col] - mu) / sigma
    return df


def add_lag_features(data, n_lags, fh=28):
    if isinstance(n_lags, int):
        lags_list = range(0, n_lags)
    for lag in lags_list:
        data[f"sales_lag_{lag + 1}"] = data.groupby("id")["sales"].transform(
            lambda x: x.shift(fh + lag)).astype("float32")
    data.dropna(inplace=True)
    lag_columns = [col for col in data.columns if col.startswith("sales_lag")]
    data.loc[:, lag_columns] = data.loc[:, lag_columns].astype("int16")


def add_rolling_features(data, windows, shifts):
    for w in windows:
        for sh in shifts:
            data[f"rolling_mean_{w}_{sh}"] = data.groupby("id")["sales"].transform(
                lambda x: x.shift(cfg.FH + sh).rolling(w).mean()).astype("float32")
            data[f"rolling_std_{w}_{sh}"] = data.groupby("id")["sales"].transform(
                lambda x: x.shift(cfg.FH + sh).rolling(w).std()).astype("float32")


def add_mean_encoding(data):
    levels = (
        ('state_id'),
        ('store_id'),
        ('cat_id'),
        ('dept_id'),
        ('state_id', 'cat_id'),
        ('state_id', 'dept_id'),
        ('store_id', 'cat_id'),
        ('store_id', 'dept_id'),
        ('item_id'),
        ('item_id', 'state_id'),
        ('item_id', 'store_id'),
    )
    for lvl in levels:
        data[f'enc_mean_{lvl}'] = data.groupby(lvl)['sales'].transform('mean').astype("float32")
        data[f'enc_std_{lvl}'] = data.groupby(lvl)['sales'].transform('std').astype("float32")


# def add_price_features(data):
#     data["price_nunique"] = prices_gby.transform("nunique").astype("int8")
#     data["item_nunique"] = data.groupby(
#         ['store_id', 'sell_price'])['item_id'].transform('nunique').astype("int16")

#     data['price_momentum'] = data['sell_price'] / data.groupby(
#         ['store_id', 'item_id'])['sell_price'].transform(lambda x: x.shift(1))
#     data['price_momentum_m'] = data['sell_price'] / data.groupby(
#         ['store_id', 'item_id', 'month'])['sell_price'].transform('mean')
#     data['price_momentum_y'] = data['sell_price'] / data.groupby(
#         ['store_id', 'item_id', 'year'])['sell_price'].transform('mean')