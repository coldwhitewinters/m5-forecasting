def get_columns(df, cond):
    return [col for col in df.columns if cond(col)]


def select_columns(df, cond):
    return df.loc[:, get_columns(df, cond)]
