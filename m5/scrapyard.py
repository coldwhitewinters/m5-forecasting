def drop_date(data):
    data.drop(columns=["date"], inplace=True)

# def prepare_agg_level(data_dir, df, lvl, cols):
#     df_agg = agg_data(df, lvl, cols)
#     id_cols = get_columns(df_agg, lambda x: x.endswith("id"))
#     df_agg.insert(0, "id", str(lvl))
#     for col in id_cols:
#         df_agg["id"] = df_agg["id"] + "-" + df_agg[col].apply(str)
#     df_agg = df_agg.drop(columns=id_cols)
#     df_agg["id"] = df_agg["id"].astype("category")
#     df_agg["d"] = df_agg["d"].astype("int16")
#     df_agg.to_parquet(data_dir / "temp" / f"df_agg_level_{lvl}.parquet")


# def prepare_all_agg_levels(data_dir, df, cols):
#     for lvl in range(1, 12 + 1):
#         prepare_agg_level(data_dir, df, lvl, cols)


# def bottom_up(data_dir, df, cols):
#     prepare_all_agg_levels(data_dir, df, cols)
#     levels = []
#     for lvl in range(1, 12 + 1):
#         df_agg = pd.read_parquet(data_dir / "temp" / f"df_agg_level_{lvl}.parquet")
#         levels.append(df_agg)
#     df_bu = pd.concat(levels)[["id", "d"] + cols]
#     return df_bu


# def prepare_train_bu(data_dir):
#     train = pd.read_parquet(data_dir / "processed/train.parquet")
#     train_bu = bottom_up(data_dir, train, ["sales"])
#     train_bu.to_parquet(data_dir / "processed/train_bu.parquet")

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

def prepare_train_val_split(data_dir, fh):
    for lvl in range(1, 12 + 1):
        for step in range(1, fh + 1):
            print(f"Splitting dataset for level {lvl} and step {step}")
            dataset = pd.read_parquet(data_dir / f"processed/datasets/{lvl}/{step}/dataset.parquet")
            N = dataset.d.max()
            train = dataset[(dataset.d <= N - fh)]
            val = dataset[(dataset.d > N - fh)]
            train.to_csv(data_dir / f"processed/datasets/{lvl}/{step}/train.csv", index=False, header=False)
            train.to_parquet(data_dir / f"processed/datasets/{lvl}/{step}/train.parquet")
            val.to_csv(data_dir / f"processed/datasets/{lvl}/{step}/val.csv", index=False, header=False)
            val.to_parquet(data_dir / f"processed/datasets/{lvl}/{step}/val.parquet")

