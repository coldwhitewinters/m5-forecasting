from pathlib import Path

# Project Directory
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Parameters
TASK = "test"
TARGET = "sales"
MULTI_STEP = True
FH = 28
N_LAGS = 14
# ROLLING_WINDOWS = [7, 14, 28, 56, 168]
# ROLLING_SHIFTS = [0, 7, 14]

PARAMS = {
    "task": "train",
    "objective": "tweedie",
    "num_iterations": 300,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 4,
    "early_stopping_round": 20,
}
