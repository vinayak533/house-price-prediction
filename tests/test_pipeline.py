import numpy as np
import pandas as pd
import pytest
from src.pipeline import (
    build_pipeline,
    build_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Provides a realistic synthetic sample matching California housing schema."""
    return pd.DataFrame(
        {
            "longitude": [-122.23, -122.22, -118.24, -121.49],
            "latitude": [37.88, 37.86, 34.05, 38.58],
            "housing_median_age": [41.0, 21.0, 52.0, 36.0],
            "total_rooms": [880.0, 7099.0, 1500.0, 2200.0],
            "total_bedrooms": [129.0, np.nan, 300.0, 450.0],  # Includes NaN
            "population": [322.0, 2401.0, 800.0, 1100.0],
            "households": [126.0, 1138.0, 280.0, 420.0],
            "median_income": [8.3252, 8.3014, 3.5000, 2.8000],
            "ocean_proximity": ["NEAR BAY", "NEAR BAY", "<1H OCEAN", "INLAND"],
            "median_house_value": [452600.0, 358500.0, 280000.0, 150000.0],
        }
    )


def test_preprocessor_structure():
    """Verify ColumnTransformer components and feature coverage."""
    preprocessor = build_preprocessor()
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_preprocessor_nan_and_oov_handling(sample_raw_data):
    """Verify that imputation and OOV categories transform without raising exceptions."""
    preprocessor = build_preprocessor()
    X = sample_raw_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    # Fit on training sample
    transformed = preprocessor.fit_transform(X)
    assert not np.isnan(transformed).any(), "Transformed output must contain no NaN values"

    # Transform unseen data with novel category and missing bedroom value
    novel_data = pd.DataFrame(
        [
            {
                "longitude": -120.0,
                "latitude": 36.0,
                "housing_median_age": 30.0,
                "total_rooms": 1000.0,
                "total_bedrooms": np.nan,
                "population": 500.0,
                "households": 200.0,
                "median_income": 4.5,
                "ocean_proximity": "UNKNOWN_VALLEY",  # Unseen OOV category
            }
        ]
    )
    novel_transformed = preprocessor.transform(novel_data)
    assert novel_transformed.shape[0] == 1
    assert not np.isnan(novel_transformed).any()


def test_end_to_end_pipeline_fit_predict(sample_raw_data):
    """Verify full Pipeline executes fit and predict cleanly."""
    pipeline = build_pipeline()
    X = sample_raw_data.drop(columns=[TARGET_COL])
    y = sample_raw_data[TARGET_COL]

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(sample_raw_data)
    assert np.all(predictions > 0), "All predicted house values should be positive"
