# Freight Rate Prediction

Predicts the price of a freight load from its distance, equipment type, weight, pickup and delivery coordinates, and date. The model is trained on `data/train-test.csv` (January -> October) and predicts the 12,000 loads in `data/validation.csv` (November–December). It also predicts a forecast for December for one fixed lane (Lexington to Fort Wayne), which `score.py` turns into a chart.

## Project structure

```
configs/
    config.yaml                   all paths and experiment settings
data/
    train-test.csv                labeled loads
    validation.csv                loads to predict
    validation_predictions.csv    empty template
    december_chart_inputs.csv     empty December file
    train-test_preprocessed.csv   created by src/pipeline.py
    validation_preprocessed.csv   created by src/pipeline.py
notebooks/
    exploration.ipynb             data checks
results/
    final_model.joblib            created by src/model.py
    validation_predictions.csv    filled by src/predict.py
    december_chart_inputs.csv     filled by src/predict.py
scorer_results/
    candidate_december.png        created by score.py
src/
    pipeline.py                   cleans the data and builds the folds
    model.py                      runs the experiments and trains the final model
    predict.py                    creates the two prediction files
    utils.py                      config loading, data loading, metrics
score.py                          the scorer script for the December file
requirements.txt
README.md
```

## Setup

```
pip install -r requirements.txt
```

The code was developed using Python 3.14. On Windows.

## How to run

Run the four commands from the project folder, in this order:

```
py -m src.pipeline
py -m src.model
py -m src.predict
py score.py --predictions results/validation_predictions.csv --december-predictions results/december_chart_inputs.csv
```

| Step | What it does | What it creates |
|---|---|---|
| `src.pipeline` | Fixes negative weights, fills missing weights and market index values | `data/train-test_preprocessed.csv`, `data/validation_preprocessed.csv` |
| `src.model` | Scores every setting in the config with time-series cross-validation and logs each one to MLflow. Then trains the best-scoring setting on all of the labeled data (January–October) and saves it | `results/final_model.joblib`, `mlflow.db` |
| `src.predict` | Loads the saved model and fills in the two prediction files | `results/validation_predictions.csv`, `results/december_chart_inputs.csv` |
| `score.py` | Draws the December chart | `scorer_results/candidate_december.png` |

The two preprocessed training and validation files are already included in `data/`. Running `src.pipeline` creates the same two files again.

## Must know before running

- **`src.predict` fills two files in place.** It reads `results/validation_predictions.csv` and `results/december_chart_inputs.csv` and writes the predictions back into them. Both files are already in the repository. If you delete them, copy the empty versions back from `data/` first, or `src.predict` will stop with a "file not found" error:
  ```
  copy data\validation_predictions.csv results\validation_predictions.csv
  copy data\december_chart_inputs.csv results\december_chart_inputs.csv
  ```
- **The `results/` folder must exist.** `src.model` saves the model there but does not create the folder.
- **Run the steps in order.** Each step uses files created by the one before it.

## Experiments

Every combination is logged to MLflow with its settings and its metrics for each fold. To look at the results:

```
py -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 in a browser.
