"""
Inference module for California House Price Prediction.
Supports programmatic invocation and CLI execution for single-record or batch CSV predictions.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Suppress loky CPU count warning on Windows platforms
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(model_path: Union[str, Path] = "model.pkl") -> Pipeline:
    """
    Loads serialized joblib Pipeline artifact.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")
    return joblib.load(path)


def predict_single(
    model: Pipeline,
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: Optional[float],
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str,
) -> float:
    """
    Executes inference for a single property specification.
    Missing total_bedrooms (np.nan or None) is handled by the internal imputer.
    """
    payload = pd.DataFrame(
        [
            {
                "longitude": float(longitude),
                "latitude": float(latitude),
                "housing_median_age": float(housing_median_age),
                "total_rooms": float(total_rooms),
                "total_bedrooms": float(total_bedrooms) if total_bedrooms is not None else np.nan,
                "population": float(population),
                "households": float(households),
                "median_income": float(median_income),
                "ocean_proximity": str(ocean_proximity),
            }
        ]
    )
    prediction = model.predict(payload)[0]
    return float(prediction)


def predict_batch(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """
    Executes inference across a batch DataFrame.
    """
    return model.predict(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="California House Price Prediction Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model.pkl",
        help="Path to serialized Pipeline (.pkl)",
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default=None,
        help="Optional CSV file path for batch inference",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions.csv",
        help="Destination path for batch inference output",
    )
    # Single sample arguments
    parser.add_argument("--longitude", type=float, default=-122.23, help="Longitude coordinate")
    parser.add_argument("--latitude", type=float, default=37.88, help="Latitude coordinate")
    parser.add_argument("--housing-median-age", type=float, default=41.0, help="Median house age")
    parser.add_argument("--total-rooms", type=float, default=880.0, help="Total rooms in block")
    parser.add_argument("--total-bedrooms", type=float, default=129.0, help="Total bedrooms (or NaN)")
    parser.add_argument("--population", type=float, default=322.0, help="Block population")
    parser.add_argument("--households", type=float, default=126.0, help="Total households in block")
    parser.add_argument("--median-income", type=float, default=8.3252, help="Median income ($10k USD)")
    parser.add_argument(
        "--ocean-proximity",
        type=str,
        default="NEAR BAY",
        choices=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
        help="Ocean proximity category",
    )

    args = parser.parse_args()

    try:
        model = load_model(args.model_path)
    except Exception as exc:
        print(f"[FATAL] Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.batch_file:
        input_path = Path(args.batch_file)
        if not input_path.exists():
            print(f"[ERROR] Batch input file does not exist: {input_path}", file=sys.stderr)
            sys.exit(1)
        data = pd.read_csv(input_path)
        predictions = predict_batch(model, data)
        data["predicted_median_house_value"] = predictions
        data.to_csv(args.output_file, index=False)
        print(f"[SUCCESS] Processed {len(data)} records. Saved predictions to {args.output_file}")
    else:
        val = predict_single(
            model=model,
            longitude=args.longitude,
            latitude=args.latitude,
            housing_median_age=args.housing_median_age,
            total_rooms=args.total_rooms,
            total_bedrooms=args.total_bedrooms,
            population=args.population,
            households=args.households,
            median_income=args.median_income,
            ocean_proximity=args.ocean_proximity,
        )
        print("=" * 50)
        print("  CALIFORNIA HOUSING PRICE PREDICTION ENGINE")
        print("=" * 50)
        print(f"  Coordinates      : ({args.latitude:.4f}, {args.longitude:.4f})")
        print(f"  Median Age       : {args.housing_median_age:.1f} years")
        print(f"  Total Rooms      : {args.total_rooms:,.0f} (Bedrooms: {args.total_bedrooms})")
        print(f"  Population       : {args.population:,.0f} | Households: {args.households:,.0f}")
        print(f"  Median Income    : ${args.median_income * 10000:,.2f}")
        print(f"  Ocean Proximity  : {args.ocean_proximity}")
        print("-" * 50)
        print(f"  PREDICTED VALUE  : ${val:,.2f} USD")
        print("=" * 50)


if __name__ == "__main__":
    main()
