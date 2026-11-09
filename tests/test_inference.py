import os
import numpy as np
import pandas as pd
import pytest
from src.predict import load_model, predict_single, predict_batch


@pytest.fixture(scope="module")
def loaded_pipeline():
    """Loads the pre-trained production model artifact."""
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        pytest.skip(f"Model artifact {model_path} not found.")
    return load_model(model_path)


def test_predict_single_expected_output(loaded_pipeline):
    """Verify single inference prediction matches benchmark expectations."""
    val = predict_single(
        model=loaded_pipeline,
        longitude=-122.230,
        latitude=37.880,
        housing_median_age=41.0,
        total_rooms=880.0,
        total_bedrooms=129.0,
        population=322.0,
        households=126.0,
        median_income=8.3252,
        ocean_proximity="NEAR BAY",
    )
    assert isinstance(val, float)
    assert 400000.0 < val < 460000.0, f"Expected value near $433k, received {val}"


def test_predict_single_with_nan_bedrooms(loaded_pipeline):
    """Verify robust inference when total_bedrooms is None/NaN."""
    val = predict_single(
        model=loaded_pipeline,
        longitude=-122.230,
        latitude=37.880,
        housing_median_age=41.0,
        total_rooms=880.0,
        total_bedrooms=None,
        population=322.0,
        households=126.0,
        median_income=8.3252,
        ocean_proximity="NEAR BAY",
    )
    assert isinstance(val, float)
    assert not np.isnan(val)
    assert val > 0


def test_predict_batch(loaded_pipeline):
    """Verify batch prediction on a sample DataFrame."""
    batch_df = pd.DataFrame(
        [
            {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY",
            },
            {
                "longitude": -118.24,
                "latitude": 34.05,
                "housing_median_age": 25.0,
                "total_rooms": 1200.0,
                "total_bedrooms": 240.0,
                "population": 600.0,
                "households": 210.0,
                "median_income": 4.1250,
                "ocean_proximity": "<1H OCEAN",
            },
        ]
    )
    preds = predict_batch(loaded_pipeline, batch_df)
    assert len(preds) == 2
    assert isinstance(preds, np.ndarray)
    assert all(p > 0 for p in preds)
