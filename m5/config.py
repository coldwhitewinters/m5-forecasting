from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_TEMP_DIR = DATA_DIR / "temp"
MODEL_DIR = ROOT_DIR / "models"

# Parameters
UNZIP = False
TARGET = "sales"
FH = 28
N_LAGS = 14
ROLLING_WINDOWS = [7, 14, 28, 56, 168]
ROLLING_SHIFTS = [0, 7, 14]