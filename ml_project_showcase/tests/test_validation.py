import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_validation.schemas import BreastCancerRecord, BreastCancerSchema
from data_validation.validation import validate_with_pydantic, validate_with_pandera


def make_valid_row():
    return {
        "Mean_Radius": 14.0, "SE_Radius": 0.5, "Worst_Radius": 16.0,
        "Mean_Texture": 19.0, "SE_Texture": 1.0, "Worst_Texture": 24.0,
        "Mean_Perimeter": 90.0, "SE_Perimeter": 5.0, "Worst_Perimeter": 110.0,
        "Mean_Area": 600.0, "SE_Area": 40.0, "Worst_Area": 800.0,
        "Mean_Smoothness": 0.1, "SE_Smoothness": 0.01, "Worst_Smoothness": 0.15,
        "Mean_Compactness": 0.2, "SE_Compactness": 0.05, "Worst_Compactness": 0.3,
        "Mean_Concavity": 0.3, "SE_Concavity": 0.05, "Worst_Concavity": 0.4,
        "Mean_ConcavePoints": 0.1, "SE_ConcavePoints": 0.02, "Worst_ConcavePoints": 0.15,
        "Mean_Symmetry": 0.2, "SE_Symmetry": 0.04, "Worst_Symmetry": 0.3,
        "Mean_FractalDimension": 0.06, "SE_FractalDimension": 0.01, "Worst_FractalDimension": 0.09,
        "Diagnosis": 1,
    }


class TestBreastCancerRecord:
    def test_valid_record(self):
        record = BreastCancerRecord(**make_valid_row())
        assert record.Diagnosis == 1
        assert record.Mean_Radius == 14.0

    def test_negative_radius_raises(self):
        data = make_valid_row()
        data["Mean_Radius"] = -1.0
        with pytest.raises(Exception):
            BreastCancerRecord(**data)

    def test_smoothness_out_of_range_raises(self):
        data = make_valid_row()
        data["Mean_Smoothness"] = 1.5
        with pytest.raises(Exception):
            BreastCancerRecord(**data)

    def test_invalid_diagnosis_raises(self):
        data = make_valid_row()
        data["Diagnosis"] = 2
        with pytest.raises(Exception):
            BreastCancerRecord(**data)


class TestValidateWithPydantic:
    def test_valid_dataframe(self):
        df = pd.DataFrame([make_valid_row(), make_valid_row()])
        valid, invalid = validate_with_pydantic(df)
        assert len(valid) == 2
        assert len(invalid) == 0

    def test_mixed_dataframe(self):
        rows = [make_valid_row()]
        bad = make_valid_row()
        bad["Mean_Radius"] = -5.0
        rows.append(bad)
        df = pd.DataFrame(rows)
        valid, invalid = validate_with_pydantic(df)
        assert len(valid) == 1
        assert len(invalid) == 1


class TestBreastCancerSchema:
    def test_valid_dataframe_passes(self):
        df = pd.DataFrame([make_valid_row(), make_valid_row()])
        success, _ = validate_with_pandera(df)
        assert success is True

    def test_invalid_dataframe_fails(self):
        df = pd.DataFrame([make_valid_row()])
        df.loc[0, "Mean_Radius"] = -5.0
        success, errors = validate_with_pandera(df)
        assert success is False
        assert errors is not None
