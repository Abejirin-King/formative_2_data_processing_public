import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transaction_simulation import (
    PARTICIPANTS, DISPLAY_NAMES, load_resources,
    extract_face_features, load_uploaded_audio, extract_audio_features,
    predict_identity
)


# Demo-only links between verified participants and existing customer records.
# These IDs are assigned for demonstration and are NOT verified personal identities.
CUSTOMER_IDS = {
    "participant_01": "100",  # Ella
    "participant_02": "101",  # King
    "participant_03": "102",  # Nicole
    "participant_04": "103",  # Sarah
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classroom command-line multimodal transaction verification."
    )
    parser.add_argument("--name", required=True, help="Claimed account: Ella, King, Nicole or Sarah")
    parser.add_argument("--face", required=True, type=Path, help="Path to a JPG or PNG facial image")
    parser.add_argument("--voice", required=True, type=Path, help="Path to a WAV voice recording")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Minimum model probability required for an identity match (default: 0.75)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("The confidence threshold must be between 0 and 1.")
    if not args.face.is_file() or not args.voice.is_file():
        raise FileNotFoundError("Both the face image and WAV recording must exist.")

    claimed_id = PARTICIPANTS.get(args.name.strip().lower())
    if claimed_id is None:
        raise ValueError("Unknown account name. Use Ella, King, Nicole or Sarah.")

    resources = load_resources()
    face_bytes = args.face.read_bytes()
    voice_bytes = args.voice.read_bytes()

    face_vector = extract_face_features(
        face_bytes, resources["detector"], resources["face_columns"]
    )
    signal = load_uploaded_audio(voice_bytes)
    audio_values = extract_audio_features(signal)
    audio_vector = pd.DataFrame(
        [[audio_values[column] for column in resources["audio_columns"]]],
        columns=resources["audio_columns"]
    )

    face_prediction, face_confidence = predict_identity(
        resources["face_model"], face_vector
    )
    voice_prediction, voice_confidence = predict_identity(
        resources["voice_model"], audio_vector
    )

    face_known = face_confidence is not None and face_confidence >= args.threshold
    voice_known = voice_confidence is not None and voice_confidence >= args.threshold
    face_prediction = face_prediction if face_known else "unknown"
    voice_prediction = voice_prediction if voice_known else "unknown"
    face_ok = face_known and face_prediction == claimed_id
    voice_ok = voice_known and voice_prediction == claimed_id

    print("\\nMULTIMODAL VERIFICATION")
    print(f"Claimed account: {DISPLAY_NAMES[claimed_id]}")
    print(f"Face result: {DISPLAY_NAMES.get(face_prediction, face_prediction)} "
          f"({face_confidence:.1%} confidence)" if face_confidence is not None
          else "Face result: unknown (model has no probability output)")
    print(f"Voice result: {DISPLAY_NAMES.get(voice_prediction, voice_prediction)} "
          f"({voice_confidence:.1%} confidence)" if voice_confidence is not None
          else "Voice result: unknown (model has no probability output)")
    print(f"Face check: {'PASS' if face_ok else 'REJECT'}")
    print(f"Voice check: {'PASS' if voice_ok else 'REJECT'}")

    if not (face_ok and voice_ok):
        print("\\nTRANSACTION REJECTED: both identity checks must match the claimed account.")
        return

    print("\\nIDENTITY VERIFIED.")
    print("The product model will now use customer transaction features associated with the verified participant.")
    customer_id = CUSTOMER_IDS.get(claimed_id)
    if customer_id is None:
        raise ValueError(
            f"No demo customer ID is configured for {DISPLAY_NAMES[claimed_id]}."
        )
    print(f"Using demo customer ID: {customer_id}")

    product_features_path = ROOT / "data" / "processed" / "tabular_model_features.csv"
    merged_path = ROOT / "data" / "processed" / "customer_transactions_social_merged.csv"
    product_data = pd.read_csv(product_features_path)
    merged_data = pd.read_csv(merged_path)
    if len(product_data) != len(merged_data):
        raise ValueError("Product feature rows do not align with the merged customer records.")

    customer_ids = merged_data["customer_id_legacy"].astype(str).str.strip()
    selected_indices = np.flatnonzero(customer_ids.to_numpy() == customer_id)
    if len(selected_indices) == 0:
        print(f"No transaction records were found for customer ID {customer_id}. Check the ID.")
        return

    excluded = {
        "product_category", "transaction_id", "customer_id_legacy",
        "customer_id_candidate", "customer_id_new", "purchase_amount",
        "customer_rating", "purchase_date"
    }
    product_X = product_data.drop(
        columns=[column for column in excluded if column in product_data.columns]
    )
    model_columns = list(resources["product_model"].feature_names_in_)
    missing_columns = sorted(set(model_columns) - set(product_X.columns))
    if missing_columns:
        raise ValueError("Product model input columns are missing: " + ", ".join(missing_columns))

    customer_rows = product_X.iloc[selected_indices][model_columns]
    prediction = resources["product_model"].predict(customer_rows)
    category = str(prediction[-1])

    print("\\nPRODUCT CATEGORY PREDICTION")
    print(f"Verified participant: {DISPLAY_NAMES[claimed_id]}")
    print(f"Associated customer ID: {customer_id}")
    print(f"Transactions used: {len(customer_rows)}")
    print(f"Predicted product category: {category}")

    if hasattr(resources["product_model"], "predict_proba"):
        probabilities = resources["product_model"].predict_proba(customer_rows)[-1]
        ranking = sorted(
            zip(resources["product_model"].classes_, probabilities),
            key=lambda item: item[1], reverse=True
        )[:3]
        print("Top category scores:")
        for label, score in ranking:
            print(f"  {label}: {score:.1%}")

    print("The customer ID must belong to the verified participant. This is an experimental category estimate.")
    print("This demo threshold is configurable and is not a security-grade calibration.")


if __name__ == "__main__":
    main()
