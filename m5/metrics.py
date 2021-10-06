import numpy as np


def rmsse(y_train, y_val, y_fcst):
    sqerr = (y_val - y_fcst)**2
    sqdiffs = y_train.diff()**2
    rmsse_value = np.sqrt(sqerr.mean() / sqdiffs.mean())
    return rmsse_value
