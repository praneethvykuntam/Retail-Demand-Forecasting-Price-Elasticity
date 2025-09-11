import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import numpy as np

REPORTS = Path("data/reports")

def main():
    df = pd.read_csv(REPORTS/"predictions.csv", parse_dates=["date"])
    mae = mean_absolute_error(df["units"], df["pred_units"])
    # Older sklearn versions don’t support squared=False; take sqrt manually.
    rmse = sqrt(mean_squared_error(df["units"], df["pred_units"]))
    print({"MAE": round(mae,3), "RMSE": round(rmse,3)})

    # Price elasticity placeholder note (needs price feature reload)
    # In a production eval, reload features to access 'price' and run log-log regression.

if __name__ == "__main__":
    main()
