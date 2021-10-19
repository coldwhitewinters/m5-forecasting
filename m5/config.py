from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
FCST_DIR = ROOT_DIR / "fcst"
METRICS_DIR = ROOT_DIR / "metrics"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Parameters
UNZIP = False
TARGET = "sales"
FH = 28
N_LAGS = 14
N_PARTITIONS = 16
# ROLLING_WINDOWS = [7, 14, 28, 56, 168]
# ROLLING_SHIFTS = [0, 7, 14]
