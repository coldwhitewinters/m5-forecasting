import m5.config as cfg

AGG_LEVEL = {
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

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

CALENDAR_FEATURES = [
    'wday', 'month', 'year', 'event_name_1', 'event_type_1',
    'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI',
]

LAG_FEATURES = [f"sales_lag_{i}" for i in range(1, cfg.N_LAGS + 1)]

if cfg.MULTI_STEP:
    STEP_RANGE = range(1, cfg.FH + 1)
else:
    STEP_RANGE = [28]
