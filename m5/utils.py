def get_columns(df, cond):
    return [col for col in df.columns if cond(col)]


def select_columns(df, cond):
    return df.loc[:, get_columns(df, cond)]


def move_column(data, col, loc=0):
    values = data[col]
    data.drop(columns=[col], inplace=True)
    data.insert(loc, col, values)


def create_dir(path):
    if not path.exists():
        path.mkdir(parents=True)
    return path
