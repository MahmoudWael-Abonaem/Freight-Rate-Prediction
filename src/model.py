import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.pipeline import build_preprocessor, make_folds
from src.utils import Root, compute_metrics, load_config, load_data

# Calculates the median price per mile.
def distance_baseline(df, cfg):
    cols = cfg["columns"]
    return (df[cols["Target"]] / df[cols["Distance"]]).median()

# Predicts the price as the price per mile times the distance.
def predict_distance_baseline(rate_per_mile, df, cfg):
    return rate_per_mile * df[cfg["columns"]["Distance"]].to_numpy(dtype=float)

# Calculates price = a * distance^b, unlike price-per-mile, it lets the cost per mile change with distance.
def power_baseline(df, cfg):
    cols = cfg["columns"]
    log_distance = np.log(df[cols["Distance"]].to_numpy(dtype=float))
    log_price = np.log(df[cols["Target"]].to_numpy(dtype=float))
    design = np.c_[np.ones(len(df)), log_distance]
    intercept, slope = np.linalg.lstsq(design, log_price, rcond=None)[0]
    return np.exp(intercept), slope

# Predicts the price as a * distance^b.
def predict_power_baseline(coefficients, df, cfg):
    a, b = coefficients
    return a * df[cfg["columns"]["Distance"]].to_numpy(dtype=float) ** b

# Builds a regressor so that it trains on the standardized log target.
def build_model(cfg, regressor, degree, include_day, target):
    use_log = target == "log"
    pipeline = make_pipeline(build_preprocessor(cfg, use_log, degree, include_day), regressor)
    if use_log:
        transformer = make_pipeline(FunctionTransformer(np.log, inverse_func=np.exp), StandardScaler())
    else:
        transformer = StandardScaler()
    return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)

# Builds a SGD linear regression model.
def build_sgd(cfg, degree, lr_setting, include_day, target):
    settings = cfg["model"]["SGD"]
    extra = {k.lower(): v for k, v in lr_setting.items() if k != "Name"}
    sgd = SGDRegressor(
                        penalty=settings["Penalty"], average=True, tol=settings["Tol"],
                        max_iter=settings["Max_Iter"], random_state=cfg["model"]["Random_State"], **extra,
                      )
    return build_model(cfg, sgd, degree, include_day, target)

# Builds a gradient boosting model.
def build_boosting(cfg, degree, learning_rate, include_day):
    boosting = HistGradientBoostingRegressor(learning_rate=learning_rate, random_state=cfg["model"]["Random_State"])
    return make_pipeline(build_preprocessor(cfg, False, degree, include_day), boosting)

# Returns the list of (run name, a function that builds a fresh model, its settings) to compare.
def model_configs(cfg):
    configs = []
    for degree in cfg["features"]["Poly_Degrees"]:
        for include_day in cfg["features"]["Include_Day"]:
            suffix = "_day" if include_day else ""
            for target in cfg["model"]["Target"]:
                for lr in cfg["model"]["SGD"]["Learning_Rates"]:
                    name = f"sgd_deg_{degree}_{target}_{lr['Name']}{suffix}"
                    settings = {
                                  "family": "sgd", "degree": degree, "target": target, "include_day": include_day,
                                  **{"lr_" + k.lower(): v for k, v in lr.items()},
                               }
                    configs.append((name, (lambda d=degree, l=lr, t=target, i=include_day: build_sgd(cfg, d, l, i, t)), settings))
                for learning_rate in cfg["model"]["Boosting"]["Learning_Rates"]:
                    name = f"boosting_deg{degree}_lr{learning_rate}{suffix}"
                    settings = {
                                "family": "boosting", "degree": degree, "target": "raw",
                                "include_day": include_day, "learning_rate": learning_rate,
                            }
                    configs.append((name, (lambda d=degree, lr=learning_rate, i=include_day: build_boosting(cfg, d, lr, i)), settings))
    return configs

# Runs the time-series cross-validation.
def run_cv(df, cfg):
    df, folds = make_folds(df, cfg)
    cols = cfg["columns"]
    y_all = df[cols["Target"]].to_numpy(dtype=float)
    configs = model_configs(cfg)
    rows = []
    for k, (train_idx, test_idx) in enumerate(folds, start=1):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        y_train = y_all[train_idx]
        predictions = {
                        "baseline_distance": predict_distance_baseline(distance_baseline(train, cfg), test, cfg),
                        "baseline_power": predict_power_baseline(power_baseline(train, cfg), test, cfg),
                      }
        for name, build, _ in configs:
            predictions[name] = build().fit(train, y_train).predict(test)
        for name, prediction in predictions.items():
            diverged = not np.all(np.isfinite(prediction))
            metrics = (
                        {"MAE": np.nan, "MAPE": np.nan, "R2": np.nan}
                        if diverged else compute_metrics(y_all[test_idx], prediction)
                      )
            rows.append({
                            "fold": k,
                            "model": name,
                            "train_rows": len(train_idx),
                            "test_start": test[cols["Date"]].min().date(),
                            "test_end": test[cols["Date"]].max().date(),
                            "diverged": diverged,
                            **metrics,
                        })
        print(f"fold {k} done")
    return pd.DataFrame(rows), configs

# Averages the metrics of every model over the folds.
def summarize(results):
    diverged = results.groupby("model")["diverged"].any()
    clean = results[~results["model"].isin(diverged[diverged].index)]
    return clean.groupby("model")[["MAE", "MAPE", "R2"]].mean().round(3)

# Points MLflow to the local SQLite database and selects the experiment.
def setup_mlflow(cfg):
    mlflow.set_tracking_uri(f"sqlite:///{Root / cfg['mlflow']['Db_Path']}")
    mlflow.set_experiment(cfg["mlflow"]["Experiment"])

# Logs one model of the CV as an MLflow run: its settings, its metrics per fold, and the mean metrics.
def log_cv_run(name, settings, results, cfg):
    rows = results[results["model"] == name]
    metrics = ["MAE", "MAPE", "R2"]
    diverged = bool(rows["diverged"].any())
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
                            **settings,
                            "n_splits": cfg["cv"]["n_splits"],
                            "gap": cfg["cv"]["Gap"],
                         })
        for _, row in rows.iterrows():
            if not row["diverged"]:
                mlflow.log_metrics({metric: float(row[metric]) for metric in metrics}, step=int(row["fold"]))
        if not diverged:
            mlflow.log_metrics({f"{metric}_mean": float(rows[metric].mean()) for metric in metrics})
        mlflow.set_tag("family", settings.get("family", "baseline_distance"))
        mlflow.set_tag("diverged", str(diverged))

# Runs the CV, logs every model to MLflow, and trains the best-scoring model on all labeled rows.
def main():
    cfg = load_config()
    df = load_data(cfg["paths"]["train_preprocessed"], cfg)

    results, configs = run_cv(df, cfg)

    print("\nMean over folds:")
    summary = summarize(results)
    print(summary.sort_values("MAE").to_string())

    setup_mlflow(cfg)
    log_cv_run("baseline_distance", {"family": "baseline_distance", "target": "none"}, results, cfg)
    log_cv_run("baseline_power", {"family": "baseline_power", "target": "none"}, results, cfg)
    for name, _, settings in configs:
        log_cv_run(name, settings, results, cfg)

    # The final model is whichever configuration has the lowest mean MAE (both baselines excluded).
    baseline_names = [n for n in ("baseline_distance", "baseline_power") if n in summary.index]
    best_name = summary.drop(baseline_names).sort_values("MAE").index[0]
    best_build = dict((name, build) for name, build, _ in configs)[best_name]
    best_settings = dict((name, settings) for name, _, settings in configs)[best_name]
    print(f"\nfinal model: {best_name} ({best_settings}), trained on all {len(df)} labeled rows")

    y = df[cfg["columns"]["Target"]].to_numpy(dtype=float)
    final_model = best_build().fit(df, y)
    model_path = Root / cfg["paths"]["final_model"]
    joblib.dump(final_model, model_path)

    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params({**best_settings, "train_rows": len(df)})
        mlflow.sklearn.log_model(sk_model=final_model, name="final_model", skops_trusted_types=["src.pipeline.day_of_week"])

if __name__ == "__main__":
    main()