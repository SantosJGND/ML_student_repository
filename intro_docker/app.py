from fastapi import FastAPI, Query, Path
from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="FastAPI Intro Demo")


class GradeInput(BaseModel):
    hours_studied: float
    attendance_rate: float
    previous_grade: float


class GradeOutput(BaseModel):
    predicted_score: float
    grade_letter: str
    message: str


class ItemOutput(BaseModel):
    item_id: int
    name: str
    price: float
    in_stock: bool


@app.get("/")
def root():
    return {
        "app": "FastAPI Intro Demo",
        "version": "1.0",
        "endpoints": {
            "GET /": "this message",
            "GET /health": "health check",
            "GET /items/{item_id}": "get an item by ID",
            "POST /predict": "predict exam score from study data",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/items/{item_id}", response_model=ItemOutput)
def get_item(
    item_id: int = Path(..., description="ID of the item"),
    min_price: Optional[float] = Query(None, description="Filter items below this price"),
):
    mock_items = {
        1: {"name": "Notebook", "price": 3.50},
        2: {"name": "Pen", "price": 1.20},
        3: {"name": "Textbook", "price": 45.00},
    }
    item = mock_items.get(item_id)
    if item is None:
        return ItemOutput(item_id=item_id, name="Unknown", price=0.0, in_stock=False)
    in_stock = True
    if min_price is not None and item["price"] > min_price:
        in_stock = False
    return ItemOutput(item_id=item_id, name=item["name"], price=item["price"], in_stock=in_stock)


@app.post("/predict", response_model=GradeOutput)
def predict_grade(data: GradeInput):
    score = (
        data.hours_studied * 2.5
        + data.attendance_rate * 5.0
        + data.previous_grade * 0.8
    )
    score = min(score, 20.0)
    score = max(score, 0.0)

    if score >= 16:
        grade = "A"
    elif score >= 14:
        grade = "B"
    elif score >= 12:
        grade = "C"
    elif score >= 10:
        grade = "D"
    else:
        grade = "F"

    return GradeOutput(
        predicted_score=round(score, 1),
        grade_letter=grade,
        message=f"Predicted score: {score:.1f}/20 — Grade: {grade}",
    )
