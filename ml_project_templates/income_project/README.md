# download link:

https://archive.ics.uci.edu/static/public/2/adult.zip

## Data Generation

### Clean Data
```bash
cd data_ingestion && python ingestion.py
```
Outputs: `data/processed/clean_train.csv`, `clean_test.csv`

### Corrupted Data
```bash
cd data_injection && python generate_corrupted.py
```
Outputs: `data/processed/corrupted/corrupted_{train|test}_{preset}.csv`

Available presets (configure in `data_injection/config.py`):
- `missing_light` / `missing_heavy` — inject NaN values
- `noise_low` / `noise_high` — gaussian noise on numeric columns
- `outliers` — artificial spike in target values
- `bias` — systematic underreporting (0.7×)
- `schema_drift` — rename target column

Modify `RANDOM_SEED` in `config.py` to regenerate with different random states.

## Scripts

The following scripts are available in the `scripts/` directory for MLOps workflows:

### Script Specifications

| Script | Required Arguments | Optional Arguments | Purpose |
|--------|-------------------|-------------------|---------|
| `run_validation.py` | `--data-path` | - | Validate data quality and schema |
| `run_training.py` | `--data-path`, `--model-type` | `--params` (JSON) | Train model with specified algorithm |
| `run_gate_evaluation.py` | `--model-uri`, `--test-data` | `--gates` (JSON) | Evaluate model against configurable gates |
| `run_registration.py` | `--model-uri`, `--model-name` | - | Register model in MLflow Model Registry |

### Supported Model Types
- `logistic` — Logistic Regression
- `xgboost` — XGBoost Classifier
- `random_forest` — Random Forest Classifier
- `svm` — Support Vector Machine

### Gate Configuration
Gates are CLI-configurable via the `--gates` argument as a JSON string. Default gates:
```json
{"accuracy": 0.95, "f1": 0.90}
```

### Usage Examples

```bash
# Validate clean training data
python scripts/run_validation.py --data-path data/processed/clean_train.csv

# Train XGBoost model with custom parameters
python scripts/run_training.py \
  --data-path data/processed/clean_train.csv \
  --model-type xgboost \
  --params '{"n_estimators": 100, "max_depth": 5}'

# Evaluate model gates (CLI-configurable)
python scripts/run_gate_evaluation.py \
  --model-uri models:/xgboost/1 \
  --test-data data/processed/clean_test.csv \
  --gates '{"accuracy": 0.95, "f1": 0.90}'

# Register model in MLflow
python scripts/run_registration.py \
  --model-uri models:/xgboost/1 \
  --model-name income_xgb
```

## Session 06: Build an App

Students build a FastAPI inference app using the templates provided.

### App Template
The `api/` directory contains the app template with TODOs for students to implement:
- `api/app.py` — FastAPI app with `/predict`, `/health`, and `/train` endpoints (TODOs)
- `models/predict.py` — Prediction functions: `predict_income()`, `predict_high_income_prob()` (TODOs)

### Example Input Files
Test the app with example JSON files in `api/example_inputs/`:
- `high_income_sample.json` — Sample with income=1 (>50K)
- `low_income_sample.json` — Sample with income=0 (<=50K)

### Usage
```bash
# Start the API server
cd income_project
uvicorn api.app:app --reload --port 8003

# Health check
curl http://localhost:8003/health

# Predict with example input (high income sample)
curl -X POST "http://localhost:8003/predict" \
  -H "Content-Type: application/json" \
  -d @api/example_inputs/high_income_sample.json
```

### Reference: Training Functions
Students should have already implemented these in `models/train.py`:
- `train_logistic_regression()`, `train_xgboost()`, `train_random_forest()`, `train_svm()`
- These can be used to (re)train models that the app will load for inference
