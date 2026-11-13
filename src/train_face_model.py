from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "images" / "processed" / "image_features.csv"
MODEL_DIR = ROOT / "models"


def main():
    if not FEATURES.exists():
        raise FileNotFoundError("Run notebooks/02_image_preprocessing.ipynb first.")

    data = pd.read_csv(FEATURES)
    required = {"source_path", "participant_id", "augmentation"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")

    feature_columns = [column for column in data if column.startswith("hog_")]
    originals = data[data["augmentation"].eq("original")]
    held_out = {
        participant: sorted(group["source_path"].unique())[-1]
        for participant, group in originals.groupby("participant_id")
    }
    train = data[~data["source_path"].isin(held_out.values())]
    if train["participant_id"].nunique() < 2:
        raise ValueError("At least two participants are required to train the face model.")

    X = train[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train["participant_id"].astype(str)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", SVC(kernel="linear", class_weight="balanced",
                           probability=True, random_state=42))
    ])
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "face_svm.joblib")
    print(f"Saved face model. Training rows: {len(train)}; held-out originals: {len(held_out)}")


if __name__ == "__main__":
    main()
