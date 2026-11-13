from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "tabular_model_features.csv"
MERGED = ROOT / "data" / "processed" / "customer_transactions_social_merged.csv"
MODEL_DIR = ROOT / "models"


def main():
    if not FEATURES.exists() or not MERGED.exists():
        raise FileNotFoundError("Run notebooks/01_tabular_preprocessing.ipynb first.")

    data = pd.read_csv(FEATURES)
    merged = pd.read_csv(MERGED)
    target = "product_category"
    group_column = "customer_id_legacy"
    if target not in data or group_column not in merged:
        raise ValueError("The target or customer grouping column is missing.")
    if len(data) != len(merged):
        raise ValueError("Feature rows and merged customer rows are not aligned.")

    excluded = {
        target, "transaction_id", "customer_id_legacy", "customer_id_candidate",
        "customer_id_new", "purchase_amount", "customer_rating", "purchase_date"
    }
    X = data.drop(columns=[column for column in excluded if column in data])
    y = data[target].astype(str)
    groups = merged[group_column].astype(str)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=42,
            min_samples_leaf=2, n_jobs=-1
        ))
    ])

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, _ = next(splitter.split(X, y, groups))
    model.fit(X.iloc[train_indices], y.iloc[train_indices])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "product_random_forest.joblib")
    print(f"Saved product model. Training rows: {len(train_indices)}; classes: {y.nunique()}")


if __name__ == "__main__":
    main()
