from pydantic import BaseModel, field_validator
from typing import Optional
import pandera as pa
from pandera.typing import DataFrame, Series


# TODO: Define Pydantic model for Iris record
# class IrisRecord(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float
#     class: int  # 0=setosa, 1=versicolor, 2=virginica
#
#     @field_validator('sepal_length')
#     def validate_sepal_length(cls, v):
#         if v <= 0 or v > 20:
#             raise ValueError('Sepal length must be positive and realistic')
#         return v
#
#     # TODO: Add validators for other fields


# TODO: Define Pandera DataFrame schema for Iris
# IrisSchema = pa.DataFrameSchema({
#     "sepal_length": pa.Column(pa.Float, checks=pa.Check.greater_than(0)),
#     "sepal_width": pa.Column(pa.Float, checks=pa.Check.greater_than(0)),
#     "petal_length": pa.Column(pa.Float, checks=pa.Check.greater_than(0)),
#     "petal_width": pa.Column(pa.Float, checks=pa.Check.greater_than(0)),
#     "class": pa.Column(pa.Int, checks=pa.Check.isin([0, 1, 2])),
# })
