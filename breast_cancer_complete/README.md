# download link:

https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip

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
- `bias` — systematic underreporting (0.7×)
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

**Gate Configuration**: Gates are CLI-configurable via `--gates` JSON. Default: `{"accuracy": 0.95, "f1": 0.90}`.

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
  --gates '{"accuracy": 0.95, "f1": 0.90}'

# Register model
python scripts/run_registration.py \
  --model-uri models:/xgboost/1 \
  --model-name xgboost_production
```

### Module Structure

Students should implement functions in these modules (function signatures provided, implementations are TODOs):

- `models/train.py` — `train_logistic()`, `train_xgboost()`, `train_random_forest()`, `train_svm()`, `train_all()`
- `models/predict.py` — `load_model()`, `predict()`, `predict_proba()`
- `evaluation/gate.py` — `check_gates()`, `evaluate_and_gate()`
- `evaluation/metrics.py` — `calculate_metrics()`, `calculate_cv_score()`

**Note**: Inference (`run_inference.py`) is out of scope for this exercise.
