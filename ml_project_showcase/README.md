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

## Docker MLOps Deployment

The project includes a full Docker-based MLOps stack with MLflow Model Registry as the single source of truth.

### Prerequisites

- Docker & Docker Compose v3.8+
- At least 2 GB RAM allocated to Docker

### Quick Start

```bash
# 1. Start services
docker-compose up -d

# 2. Seed MLflow Registry with pre-trained models
docker-compose run app python mlops/bootstrap.py

# 3. Verify API
curl http://localhost:8000/health
```

### Services

| Service | Port | Description |
|---|---|---|
| `mlflow` | 5000 | MLflow Tracking Server (SQLite backend, Model Registry) |
| `app` | 8000 | FastAPI inference API |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health + model registry status |
| `POST` | `/predict` | Predict using Production model from MLflow Registry |
| `GET` | `/metrics` | Request count, avg latency, current model version |
| `POST` | `/train` | Train, log to MLflow, register model |

### Execution Modes

The `app` container supports three modes via command override:

```bash
# Mode A — API server (default)
docker-compose up -d

# Mode B — Training pipeline
docker-compose run app python mlops/train_pipeline.py

# Mode C — Evaluation gate
docker-compose run app python mlops/gate.py --model-type logistic --run-id <run_id>
```

### Workflow

```bash
# Train all 4 models → register → gate → promote to Staging
docker-compose run app python mlops/train_pipeline.py

# Promote a model from Staging → Production (via MLflow UI)
open http://localhost:5000

# The API automatically serves the latest Production model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Mean_Radius": 14.0, ..., "model": "logistic"}'
```

### Utility Scripts

```bash
# Reset MLflow state (delete all models, experiments, monitoring logs)
docker-compose run app python mlops/cleandb.py

# Re-seed registry with existing .joblib models
docker-compose run app python mlops/bootstrap.py
```

### Architecture Rules

- The API **never** loads local models — only MLflow Registry (`models:/{name}/Production` or `Staging`)
- Model naming: `{project_name}_{model_type}` (e.g., `breast_cancer_logistic`)
- Stages: `None → Staging` (auto on gate pass) → `Production` (manual via MLflow UI)
- Monitoring logs: `logs/predictions.csv` (timestamp, model version, prediction, latency)
