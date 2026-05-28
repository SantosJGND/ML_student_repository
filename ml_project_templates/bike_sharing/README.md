# download link:

https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip

## Data Generation

### Clean Data
Generate processed clean data from raw files:
```bash
cd data_ingestion && python ingestion.py
```
Output: `data/processed/clean_data.csv`

### Corrupted Data
Generate corrupted datasets with configurable error types:
```bash
cd data_injection && python generate_corrupted.py
```
Output: `data/processed/corrupted/corrupted_{preset}.csv`

Available presets (configure in `data_injection/config.py`):
- `missing_light` / `missing_heavy` — inject NaN values
- `noise_low` / `noise_high` — gaussian noise on numeric columns
- `outliers` — artificial spike in target values
- `bias` — systematic underreporting (0.7x)
- `schema_drift` — rename target column

Modify `RANDOM_SEED` in `config.py` to regenerate with different random states.

## Model Development Scripts

### Script Specifications

Scripts are located in the `scripts/` directory. All scripts have full argparse arguments implemented, but the core logic is left as TODOs for students to implement.

| Script | Arguments | Description |
|---|---|---|
| `run_validation.py` | `--data-path` (required) | Validate any input data (clean or corrupted) using Pydantic and Pandera |
| `run_training.py` | `--data-path`, `--model-type`, `--params` (JSON) | Train specified model type with optional hyperparameters |
| `run_gate_evaluation.py` | `--model-uri`, `--test-data`, `--gates` (JSON) | Evaluate model against CLI-configurable gates (pass/fail) |
| `run_registration.py` | `--model-uri`, `--model-name` | Register model in MLflow Model Registry |

**Gate Configuration**: Gates are CLI-configurable via `--gates` JSON. Default: `{"mae": 300, "r2": 0.7}`.

### Example Usage

```bash
# Validate clean data
python scripts/run_validation.py --data-path data/processed/clean_data.csv

# Validate corrupted data
python scripts/run_validation.py --data-path data/processed/corrupted/corrupted_missing_light.csv

# Train XGBoost with custom params
python scripts/run_training.py \
  --data-path data/processed/clean_data.csv \
  --model-type xgboost \
  --params '{"n_estimators": 100, "max_depth": 5}'

# Evaluate gates (CLI-configurable)
python scripts/run_gate_evaluation.py \
  --model-uri models:/xgboost/1 \
  --test-data data/processed/clean_data.csv \
  --gates '{"mae": 300, "r2": 0.7}'

# Register model
python scripts/run_registration.py \
  --model-uri models:/xgboost/1 \
  --model-name xgboost_production
```

### Module Structure

Students should implement functions in these modules (function signatures provided, implementations are TODOs):

- `models/train.py` -- `train_linear_regression()`, `train_xgboost()`, `train_random_forest()`, `train_svr()`, `train_all()`
- `models/predict.py` -- `load_model()`, `predict()`, `predict_bike_demand()`
- `evaluation/gate.py` -- `check_gates()`, `evaluate_and_gate()`
- `evaluation/metrics.py` -- `calculate_metrics()`, `calculate_cv_score()`

**Note**: Inference (`run_inference.py`) is out of scope for this exercise.

## Session 06: Build an App

Students build a FastAPI inference app using the templates provided.

### App Template
The `api/` directory contains the app template with TODOs for students to implement:
- `api/app.py` -- FastAPI app with `/predict` and `/health` endpoints (TODOs)
- `models/predict.py` -- Prediction function: `predict_bike_demand()` (TODO)

### Example Input Files
Test the app with example JSON files in `api/example_inputs/`:
- `high_demand_sample.json` -- Sample with high bike demand (summer evening)
- `low_demand_sample.json` -- Sample with low bike demand (winter night)

### Usage
```bash
# Start the API server
cd bike_sharing
uvicorn api.app:app --reload --port 8005

# Health check
curl http://localhost:8005/health

# Predict with example input (high demand)
curl -X POST "http://localhost:8005/predict" \
  -H "Content-Type: application/json" \
  -d @api/example_inputs/high_demand_sample.json

# Predict with example input (low demand)
curl -X POST "http://localhost:8005/predict" \
  -H "Content-Type: application/json" \
  -d @api/example_inputs/low_demand_sample.json
```

### Reference: Training Functions
Students should have already implemented these in `models/train.py`:
- `train_linear_regression()`, `train_xgboost()`, `train_random_forest()`, `train_svr()`
- These can be used to (re)train models that the app will load for inference
