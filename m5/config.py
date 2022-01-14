from pathlib import Path

# Project Directory
ROOT_DIR = Path(__file__).parent.parent.resolve()

# General Parameters
TASK = "test"
TARGET = "sales"
FH = 28

# LGBM Parameters
N_LAGS = 14
PARAMS = {
    "task": "train",
    "objective": "tweedie",
    "num_iterations": 300,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 4,
    "early_stopping_round": 20,
}
