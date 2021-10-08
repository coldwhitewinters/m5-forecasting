def build_lag_features(df, target, fh, n_lags):
    for lag in range(0, n_lags):
        df[f"{target}_lag_{lag + 1}"] = df["sales"].shift(fh + lag).astype("float32")
    df.dropna(inplace=True)
    lag_columns = [col for col in df.columns if col.startswith("sales_lag")]
    df[lag_columns] = df[lag_columns].astype("int32")
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
