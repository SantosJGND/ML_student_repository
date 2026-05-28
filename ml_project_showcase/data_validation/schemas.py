from pydantic import BaseModel, Field, field_validator
import pandera.pandas as pa


class BreastCancerRecord(BaseModel):
    Mean_Radius: float = Field(ge=0)
    SE_Radius: float = Field(ge=0)
    Worst_Radius: float = Field(ge=0)
    Mean_Texture: float = Field(ge=0)
    SE_Texture: float = Field(ge=0)
    Worst_Texture: float = Field(ge=0)
    Mean_Perimeter: float = Field(ge=0)
    SE_Perimeter: float = Field(ge=0)
    Worst_Perimeter: float = Field(ge=0)
    Mean_Area: float = Field(ge=0)
    SE_Area: float = Field(ge=0)
    Worst_Area: float = Field(ge=0)
    Mean_Smoothness: float = Field(ge=0)
    SE_Smoothness: float = Field(ge=0)
    Worst_Smoothness: float = Field(ge=0)
    Mean_Compactness: float = Field(ge=0)
    SE_Compactness: float = Field(ge=0)
    Worst_Compactness: float = Field(ge=0)
    Mean_Concavity: float = Field(ge=0)
    SE_Concavity: float = Field(ge=0)
    Worst_Concavity: float = Field(ge=0)
    Mean_ConcavePoints: float = Field(ge=0)
    SE_ConcavePoints: float = Field(ge=0)
    Worst_ConcavePoints: float = Field(ge=0)
    Mean_Symmetry: float = Field(ge=0)
    SE_Symmetry: float = Field(ge=0)
    Worst_Symmetry: float = Field(ge=0)
    Mean_FractalDimension: float = Field(ge=0)
    SE_FractalDimension: float = Field(ge=0)
    Worst_FractalDimension: float = Field(ge=0)
    Diagnosis: int = Field(ge=0, le=1)

    @field_validator("Mean_Smoothness", "SE_Smoothness", "Worst_Smoothness",
                     "Mean_Compactness", "SE_Compactness", "Worst_Compactness",
                     "Mean_Concavity", "SE_Concavity", "Worst_Concavity",
                     "Mean_ConcavePoints", "SE_ConcavePoints", "Worst_ConcavePoints",
                     "Mean_Symmetry", "SE_Symmetry", "Worst_Symmetry",
                     "Mean_FractalDimension", "SE_FractalDimension", "Worst_FractalDimension")
    @classmethod
    def check_ratio_range(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"Value must be between 0 and 1, got {v}")
        return v


BreastCancerSchema = pa.DataFrameSchema(
    columns={
        "Mean_Radius": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "SE_Radius": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "Worst_Radius": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "Mean_Texture": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "SE_Texture": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "Worst_Texture": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "Mean_Perimeter": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "SE_Perimeter": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "Worst_Perimeter": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "Mean_Area": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "SE_Area": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "Worst_Area": pa.Column(float, pa.Check(lambda s: s >= 0, element_wise=False)),
        "Mean_Smoothness": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "SE_Smoothness": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Worst_Smoothness": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Mean_Compactness": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "SE_Compactness": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Worst_Compactness": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Mean_Concavity": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "SE_Concavity": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Worst_Concavity": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Mean_ConcavePoints": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "SE_ConcavePoints": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Worst_ConcavePoints": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Mean_Symmetry": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "SE_Symmetry": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Worst_Symmetry": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Mean_FractalDimension": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "SE_FractalDimension": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Worst_FractalDimension": pa.Column(float, pa.Check(lambda s: (s >= 0) & (s <= 1), element_wise=False)),
        "Diagnosis": pa.Column(int, pa.Check(lambda s: s.isin([0, 1]), element_wise=False)),
    },
)
