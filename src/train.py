"""
Training module for California House Price Prediction pipeline.
Handles dataset loading, train/test partitioning, cross-validation, hyperparameter tuning,
and serialization of the end-to-end ColumnTransformer + HistGradientBoostingRegressor artifact.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split, GridSearchCV

from src.pipeline import (
    TARGET_COL,
    build_pipeline,
    DEFAULT_HGB_PARAMS,
)


def load_dataset(csv_path: str = "housing.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads dataset and splits features (X) and target (y).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from {path}")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def train_and_evaluate(
    data_path: str = "housing.csv",
    output_model_path: str = "model.pkl",
    tune: bool = False,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Executes training workflow and persists the trained pipeline.
    """
    print(f"[INFO] Loading dataset from {data_path}...")
    X, y = load_dataset(data_path)
    print(f"[INFO] Total records: {len(X):,} | Feature columns: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    print(f"[INFO] Training set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    if tune:
        print("[INFO] Initiating GridSearchCV hyperparameter optimization...")
        base_pipe = build_pipeline()
        param_grid = {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [None, 3, 6],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        grid = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print(f"[INFO] Best CV RMSE: {-grid.best_score_:.2f}")
        print(f"[INFO] Best Parameters: {grid.best_params_}")
        best_pipeline = grid.best_estimator_
    else:
        print("[INFO] Building pipeline with tuned production defaults...")
        best_pipeline = build_pipeline(DEFAULT_HGB_PARAMS)
        best_pipeline.fit(X_train, y_train)

    train_preds = best_pipeline.predict(X_train)
    test_preds = best_pipeline.predict(X_test)

    metrics = {
        "train_rmse": float(root_mean_squared_error(y_train, train_preds)),
        "train_mae": float(mean_absolute_error(y_train, train_preds)),
        "train_r2": float(r2_score(y_train, train_preds)),
        "test_rmse": float(root_mean_squared_error(y_test, test_preds)),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
    }

    print("\n" + "=" * 55)
    print("           MODEL EVALUATION SUMMARY REPORT")
    print("=" * 55)
    print(f"  Train RMSE : ${metrics['train_rmse']:>10,.2f}  |  Test RMSE : ${metrics['test_rmse']:>10,.2f}")
    print(f"  Train MAE  : ${metrics['train_mae']:>10,.2f}  |  Test MAE  : ${metrics['test_mae']:>10,.2f}")
    print(f"  Train R2   : {metrics['train_r2']:>11.4f}  |  Test R2   : {metrics['test_r2']:>11.4f}")
    print("=" * 55)

    joblib.dump(best_pipeline, output_model_path)
    print(f"[SUCCESS] Serialized pipeline persisted to: {Path(output_model_path).resolve()}\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="California House Price Prediction Training Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=str, default="housing.csv", help="Path to raw CSV dataset")
    parser.add_argument("--output-model", type=str, default="model.pkl", help="Destination artifact path")
    parser.add_argument("--tune", action="store_true", help="Execute hyperparameter grid search")
    parser.add_argument("--random-state", type=int, default=42, help="Deterministic PRNG seed")

    args = parser.parse_args()
    train_and_evaluate(
        data_path=args.data_path,
        output_model_path=args.output_model,
        tune=args.tune,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
