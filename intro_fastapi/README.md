# FastAPI — Introduction

FastAPI is a modern Python web framework for building APIs. It is fast, easy to use, and automatically generates interactive documentation.

## Why FastAPI for ML?

Every project in this course uses a FastAPI API to serve model predictions. Understanding FastAPI means you can:

- Serve your trained models behind a real HTTP endpoint
- Let other applications (dashboards, web apps, mobile apps) talk to your model
- Use the automatically generated `/docs` page to test your API without writing a single client

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --reload
```

- `app:app` — `app.py` file, `app` variable (the FastAPI object)
- `--reload` — auto-restart when you edit the code (development only)

Open your browser at **http://127.0.0.1:8000**

## Explore the API

### 1. Root endpoint

```
GET http://127.0.0.1:8000/
```

Returns a welcome message with available endpoints.

### 2. Health check

```
GET http://127.0.0.1:8000/health
```

Returns `{"status": "healthy"}`. Every ML project API in this course has a `/health` endpoint.

### 3. Item lookup (path + query params)

```
GET http://127.0.0.1:8000/items/1
GET http://127.0.0.1:8000/items/2?min_price=2.0
```

- `item_id` in the URL path — **path parameter**
- `min_price` after `?` — **query parameter** (optional)

### 4. Prediction (POST with JSON body)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"hours_studied": 4.0, "attendance_rate": 0.9, "previous_grade": 12.0}'
```

Returns:

```json
{
  "predicted_score": 19.1,
  "grade_letter": "A",
  "message": "Predicted score: 19.1/20 — Grade: A"
}
```

This uses a **Pydantic model** for input validation and a **response model** for typed output — the same pattern used in all course APIs.

## Automatic Documentation

Visit **http://127.0.0.1:8000/docs** — FastAPI generates a Swagger UI where you can:

- See all endpoints
- Read parameter types and descriptions
- Click **Try it out** to send real requests from the browser

There is also an alternative ReDoc UI at **http://127.0.0.1:8000/redoc**.

## Try it in Python

```python
import requests

# Check health
r = requests.get("http://127.0.0.1:8000/health")
print(r.json())                    # {'status': 'healthy'}

# Predict a grade
data = {"hours_studied": 3.5, "attendance_rate": 0.8, "previous_grade": 11.0}
r = requests.post("http://127.0.0.1:8000/predict", json=data)
print(r.json())
# {'predicted_score': 16.6, 'grade_letter': 'A', 'message': '...'}
```

## Code Walkthrough

### 1. Imports and App Creation

```python
from fastapi import FastAPI, Query, Path
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="FastAPI Intro Demo")
```

- `FastAPI()` creates the application instance
- `title` appears in the auto-generated docs

### 2. Pydantic Models

```python
class GradeInput(BaseModel):
    hours_studied: float
    attendance_rate: float
    previous_grade: float


class GradeOutput(BaseModel):
    predicted_score: float
    grade_letter: str
    message: str
```

- Pydantic models define the **shape** of request / response data
- FastAPI automatically validates incoming JSON against these models
- If the client sends wrong types, FastAPI returns a 422 error with details

### 3. Path Operation (`@app.get`, `@app.post`)

```python
@app.get("/health")
def health():
    return {"status": "healthy"}
```

- `@app.get` — handle GET requests
- `@app.post` — handle POST requests
- The function return value is automatically converted to JSON

### 4. Path and Query Parameters

```python
@app.get("/items/{item_id}")
def get_item(
    item_id: int = Path(..., description="ID of the item"),
    min_price: Optional[float] = Query(None),
):
    ...
```

- `{item_id}` in the URL path → path parameter
- `min_price` after `?` → query parameter
- `Path` and `Query` add validation and documentation

### 5. Request Body (POST)

```python
@app.post("/predict", response_model=GradeOutput)
def predict_grade(data: GradeInput):
    ...
    return GradeOutput(...)
```

- `data: GradeInput` tells FastAPI to parse the JSON body into a Pydantic model
- `response_model=GradeOutput` tells FastAPI what shape to return (and documents it in Swagger)

## Key Concepts Summary

| Concept | What it means | Used in |
|---------|---------------|---------|
| **Path operation** | A function that handles a specific HTTP method + URL path | `@app.get`, `@app.post` |
| **Path parameter** | Variable in the URL path (`{item_id}`) | Item lookup |
| **Query parameter** | Key-value after `?` (`?name=value`) | Optional filters |
| **Request body** | JSON data sent with POST/PUT | `/predict` |
| **Pydantic model** | A class that defines data shape + validation | Input/output schemas |
| **Response model** | Guarantees the response matches a schema | All POST endpoints |
| **`/docs`** | Auto-generated Swagger UI | Testing without curl |

## Relation to Course APIs

The ML project APIs you will work with (breast_cancer, heart_disease, etc.) follow the exact same structure:

```
POST /predict  ← JSON input → Pydantic model → model.predict() → JSON output
GET  /health   ← Always returns {"status": "healthy"}
POST /train    ← Triggers model training pipeline
```

The only differences are:
- The Pydantic models have 10-70 fields (all the features)
- The prediction logic uses a real ML model instead of a formula

## Next Steps

1. Modify `app.py` — add your own endpoint
2. Add a new Pydantic model with different field types (`int`, `bool`, `list`)
3. Look at the `/docs` page to see your changes reflected instantly
4. Check the `breast_cancer_copy/api/app.py` reference to see a real ML API
