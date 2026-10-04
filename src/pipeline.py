import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler)

from src.utils import Root, load_config, load_data

# pandas' dt.dayofweek numbers the days 0 to 6, starting from Monday: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
Day_Numbers = list(range(7))

# Replaces negative weights with their absolute value.
def fix_negative_weight(df, cfg):
    weight = cfg["columns"]["Weight"]
    df[weight] = df[weight].abs()
    return df

# Fills missing weights with the column median.
def fill_weight(df, cfg, weight_median):
    weight = cfg["columns"]["Weight"]
    df[weight] = df[weight].fillna(weight_median)
    return df

# Fills missing market index with the median market index of the same day.
def fill_market_index(df, cfg):
    date = cfg["columns"]["Date"]
    market_index = cfg["columns"]["Market_Index"]
    date_median = df.groupby(date)[market_index].transform("median")
    df[market_index] = df[market_index].fillna(date_median)
    return df

# Applies all fixes to one file.
def preprocess(df, cfg, weight_median):
    df = df.copy()
    df = fix_negative_weight(df, cfg)
    df = fill_weight(df, cfg, weight_median)
    df = fill_market_index(df, cfg)
    return df

# Sorts the data by date and returns it with the (train rows, test rows) positions of every fold.
def make_folds(df, cfg):
    date = cfg["columns"]["Date"]
    df = df.sort_values(date, kind="stable").reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=cfg["cv"]["n_splits"], gap=cfg["cv"]["Gap"])
    return df, list(splitter.split(df))

# Turns one date column into its day-of-week number.
def day_of_week(frame):
    return pd.to_datetime(frame.iloc[:, 0]).dt.dayofweek.to_numpy().reshape(-1, 1)

# Builds the sklearn transformer: distance polynomial (of the given degree), scaled numeric columns,
# one-hot equipment, and, only if include_day is True, a one-hot day-of-week feature built straight
# from the date column.
def build_preprocessor(cfg, use_log, degree, include_day=False):
    cols = cfg["columns"]
    distance_steps = [FunctionTransformer(np.log, feature_names_out="one-to-one")] if use_log else []
    distance_steps += [StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False), StandardScaler()]
    steps = [
                ("distance", make_pipeline(*distance_steps), [cols["Distance"]]),
                ("numeric", StandardScaler(), [cols[key] for key in cfg["features"]["Numeric"]]),
                ("equipment", OneHotEncoder(categories=[cfg["features"]["Equipment_Categories"]], sparse_output=False), [cols["Equipment"]]),
            ]
    if include_day:
        day_pipeline = make_pipeline(
                                        FunctionTransformer(day_of_week, feature_names_out="one-to-one"),
                                        OneHotEncoder(categories=[Day_Numbers], sparse_output=False),
                                    )
        steps.append(("day_of_week", day_pipeline, [cols["Date"]]))
    return ColumnTransformer(steps)

# Loads train and validation, cleans both, and saves them as CSV files.
def main():
    cfg = load_config()
    weight = cfg["columns"]["Weight"]
    market_index = cfg["columns"]["Market_Index"]

    train = load_data(cfg["paths"]["train"], cfg)
    val = load_data(cfg["paths"]["validation"], cfg)

    # Learned on train only, after the sign fix, then reused for validation.
    weight_median = train[weight].abs().median()
    print(f"weight median (train): {weight_median}")

    for df, name, out in [
                            (train, "train", cfg["paths"]["train_preprocessed"]),
                            (val, "validation", cfg["paths"]["validation_preprocessed"]),
                         ]:
        clean = preprocess(df, cfg, weight_median)
        clean.to_csv(Root / out, index=False)
        print(f"[{name}] saved {clean.shape} -> {out}")
        print(f"[{name}] missing weight: {clean[weight].isna().sum()}, missing market_index: {clean[market_index].isna().sum()}, negative weight: {(clean[weight] < 0).sum()}")

if __name__ == "__main__":
    main()