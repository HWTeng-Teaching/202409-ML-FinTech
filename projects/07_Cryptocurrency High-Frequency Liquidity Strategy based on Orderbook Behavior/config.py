# config.py
import os
from datetime import datetime, timedelta

# Configuration parameters
BINANCE_BASE_URL = "https://api.binance.com"
DEPTH_ENDPOINT = "/api/v3/depth"
KLINES_ENDPOINT = "/api/v3/klines"

# Data parameters
SYMBOL = "BTCUSDT"
LIMIT = 8000  # Number of samples
LOOKBACK_WINDOW = 10  # Number of previous observations to use
TRAIN_SIZE = 0.7
TEST_SIZE = 0.3

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
# TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
TIMESTAMP = str(20241123161931)

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, TIMESTAMP), exist_ok=True)