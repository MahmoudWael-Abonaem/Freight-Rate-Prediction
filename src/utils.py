import yaml
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from pathlib import Path

Root = Path(__file__).resolve().parent.parent

# Reads config.yaml and returns it as a dictionary.
def load_config():
    with open(Root / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)

# Reads a CSV file and converts the date column from plain text into real dates.
def load_data(path, cfg):
    return pd.read_csv(Root / path, parse_dates=[cfg["columns"]["Date"]])

# Computes the MAE, the MAPE and the R2 metrics.
def compute_metrics(y_true, y_pred):
    return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "R2": r2_score(y_true, y_pred),
          }
