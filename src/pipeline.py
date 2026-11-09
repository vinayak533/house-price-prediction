"""
Pipeline definition module for California House Price Prediction.
Encapsulates feature transformations, preprocessing, and model assembly.
"""

from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET_COL = "median_house_value"

NUMERICAL_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

CATEGORICAL_FEATURES = [
    "ocean_proximity",
]

DEFAULT_HGB_PARAMS: Dict[str, Any] = {
    "l2_regularization": 0.1,
    "learning_rate": 0.1,
    "max_depth": None,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "random_state": 42,
}


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an isolated, leakage-free ColumnTransformer for numerical and categorical streams.
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Constructs the end-to-end Pipeline with preprocessing and HistGradientBoostingRegressor.
    """
    params = DEFAULT_HGB_PARAMS.copy()
    if model_params:
        params.update(model_params)

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", HistGradientBoostingRegressor(**params)),
        ]
    )
