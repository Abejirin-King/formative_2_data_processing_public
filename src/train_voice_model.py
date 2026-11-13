from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "audio" / "processed" / "audio_features.csv"
MODEL_DIR = ROOT / "models"


def main():
    if not FEATURES.exists():
        raise FileNotFoundError("Run notebooks/03_audio_preprocessing.ipynb first.")

    data = pd.read_csv(FEATURES)
    required = {"source_path", "participant_id", "phrase", "augmentation"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")

    metadata = required | {"audio_path"}
    feature_columns = [
        column for column in data.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(data[column])
    ]
    originals = data[data["augmentation"].eq("original")]
    train_sources = []
    for participant, group in originals.groupby("participant_id"):
        phrase_map = group.groupby("phrase")["source_path"].first().to_dict()
        if "yes_approve" not in phrase_map:
            raise ValueError(f"{participant} is missing the yes_approve recording.")
        train_sources.append(phrase_map["yes_approve"])

    train = data[data["source_path"].isin(train_sources)]
    X = train[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train["participant_id"].astype(str)
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42,
        min_samples_leaf=1, n_jobs=-1
    )
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "voice_random_forest.joblib")
    print(f"Saved voice model. Training rows: {len(train)}; participants: {y.nunique()}")


if __name__ == "__main__":
    main()
