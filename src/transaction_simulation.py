from io import BytesIO
from math import gcd
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from skimage.feature import hog
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
IMAGE_FEATURES = DATA / "images" / "processed" / "image_features.csv"
AUDIO_FEATURES = DATA / "audio" / "processed" / "audio_features.csv"
PRODUCT_FEATURES = PROCESSED / "tabular_model_features.csv"
MERGED_DATA = PROCESSED / "customer_transactions_social_merged.csv"
MODEL_DIR = ROOT / "models"
CASCADE_PATH = DATA / "models" / "haarcascade_frontalface_default.xml"
TARGET_SR = 16000
CONFIDENCE_THRESHOLD = 0.60

PARTICIPANTS = {
    "ella": "participant_01",
    "king": "participant_02",
    "nicole": "participant_03",
    "sarah": "participant_04",
    "participant_01": "participant_01",
    "participant_02": "participant_02",
    "participant_03": "participant_03",
    "participant_04": "participant_04",
}
DISPLAY_NAMES = {
    "participant_01": "Ella",
    "participant_02": "King",
    "participant_03": "Nicole",
    "participant_04": "Sarah",
}

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multimodal Transaction Verification</title>
<style>
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 36px 18px; background: #f3f5f8; color: #182230;
       font-family: Arial, sans-serif; }
main { max-width: 760px; margin: auto; }
.panel { background: white; border: 1px solid #e0e5eb; border-radius: 16px;
         padding: 28px; box-shadow: 0 8px 28px rgba(20,35,55,.06); }
h1 { margin: 0 0 8px; font-size: 27px; }
.sub { color: #596579; line-height: 1.5; margin: 0 0 24px; }
label { display: block; font-weight: 700; margin: 18px 0 8px; }
input[type=text], input[type=file] { width: 100%; padding: 12px; border: 1px solid #cbd3dd;
                                      border-radius: 9px; background: white; }
button { width: 100%; margin-top: 24px; padding: 14px; border: 0; border-radius: 9px;
         background: #175cd3; color: white; font-size: 16px; font-weight: 700; cursor: pointer; }
button:hover { background: #1249ad; }
.notice { margin-top: 18px; padding: 13px 15px; border-radius: 9px; line-height: 1.5; }
.error { background: #fff0f0; color: #a12626; }
.success { background: #eaf7ef; color: #17643a; }
.failure { background: #fff0f0; color: #a12626; }
.result { margin-top: 24px; border-top: 1px solid #e5e9ef; padding-top: 20px; }
.result h2 { font-size: 19px; margin: 0 0 12px; }
.result p { margin: 9px 0; }
.small { font-size: 13px; color: #687386; line-height: 1.5; margin-top: 20px; }
.tag { display: inline-block; min-width: 58px; padding: 4px 8px; border-radius: 6px;
       font-size: 12px; font-weight: 700; text-align: center; }
.pass { background: #e5f5eb; color: #17643a; }
.fail { background: #ffebeb; color: #a12626; }
</style>
</head>
<body>
<main>
<section class="panel">
<h1>Multimodal Transaction Verification</h1>
<p class="sub">Enter the account holder's name, upload a facial image and provide a WAV voice sample. The transaction proceeds to product-category prediction only when both identity checks match.</p>
<form method="post" enctype="multipart/form-data">
<label for="name">Account holder name</label>
<input id="name" name="name" type="text" placeholder="Enter Ella, King, Nicole or Sarah" required value="{{ name or '' }}">
<label for="face">Facial image</label>
<input id="face" name="face" type="file" accept=".jpg,.jpeg,.png,.bmp,.webp,image/*" required>
<label for="voice">Voice sample (WAV)</label>
<input id="voice" name="voice" type="file" accept=".wav,audio/wav" required>
<button type="submit">Verify and process transaction</button>
</form>
{% if error %}
<div class="notice error">{{ error }}</div>
{% endif %}
{% if result %}
<div class="result">
<h2>Verification result</h2>
<p><strong>Claimed account:</strong> {{ result.claimed_name }}</p>
<p><strong>Face model identified:</strong> {{ result.face_name }} ({{ result.face_confidence }})
<span class="tag {{ 'pass' if result.face_ok else 'fail' }}">{{ 'PASS' if result.face_ok else 'FAIL' }}</span></p>
<p><strong>Voice model identified:</strong> {{ result.voice_name }} ({{ result.voice_confidence }})
<span class="tag {{ 'pass' if result.voice_ok else 'fail' }}">{{ 'PASS' if result.voice_ok else 'FAIL' }}</span></p>
{% if result.authorized %}
<div class="notice success"><strong>Transaction authorized.</strong><br>
Product-category suggestion: <strong>{{ result.category }}</strong></div>
<p class="small">Top product-model scores: {{ result.top_scores }}</p>
{% else %}
<div class="notice failure"><strong>Transaction rejected.</strong> Both biometric identity checks must match the claimed account.</div>
{% endif %}
</div>
{% endif %}
<p class="small">Classroom demonstration only. Uploaded files are processed in memory and are not saved by this application. This prototype does not include liveness detection or anti-spoofing and must not be used for real transaction security.</p>
</section>
</main>
</body>
</html>
"""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def load_resources():
    required = [
        IMAGE_FEATURES,
        AUDIO_FEATURES,
        PRODUCT_FEATURES,
        MERGED_DATA,
        MODEL_DIR / "face_svm.joblib",
        MODEL_DIR / "voice_random_forest.joblib",
        MODEL_DIR / "product_random_forest.joblib",
        CASCADE_PATH,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required project files are missing:\n- "
            + "\n- ".join(missing)
            + "\nRun notebooks 01-04 first and make sure the Haar cascade file is available."
        )

    face_columns = [
        column for column in pd.read_csv(IMAGE_FEATURES, nrows=1).columns
        if column.startswith("hog_")
    ]
    audio_frame = pd.read_csv(AUDIO_FEATURES, nrows=1)
    audio_metadata = {"audio_path", "source_path", "participant_id", "phrase", "augmentation"}
    audio_columns = [
        column for column in audio_frame.columns
        if column not in audio_metadata and pd.api.types.is_numeric_dtype(audio_frame[column])
    ]

    face_model = joblib.load(MODEL_DIR / "face_svm.joblib")
    voice_model = joblib.load(MODEL_DIR / "voice_random_forest.joblib")
    product_model = joblib.load(MODEL_DIR / "product_random_forest.joblib")
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"Could not load the Haar cascade file: {CASCADE_PATH}")

    product_df = pd.read_csv(PRODUCT_FEATURES)
    merged_df = pd.read_csv(MERGED_DATA)
    target = "product_category"
    group_column = "customer_id_legacy"
    excluded = {
        target, "transaction_id", "customer_id_legacy", "customer_id_candidate",
        "customer_id_new", "purchase_amount", "customer_rating", "purchase_date"
    }
    if target not in product_df or group_column not in merged_df:
        raise ValueError("Product data is missing the target or customer grouping column.")
    if len(product_df) != len(merged_df):
        raise ValueError("Product features and merged customer rows are not aligned.")

    product_x = product_df.drop(columns=[column for column in excluded if column in product_df])
    product_y = product_df[target].astype(str)
    groups = merged_df[group_column].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, test_indices = next(splitter.split(product_x, product_y, groups))
    product_demo_rows = product_x.iloc[test_indices].reset_index(drop=True)

    return {
        "face_model": face_model,
        "voice_model": voice_model,
        "product_model": product_model,
        "face_columns": face_columns,
        "audio_columns": audio_columns,
        "detector": detector,
        "product_demo_rows": product_demo_rows,
    }


def extract_face_features(file_bytes, detector, feature_columns):
    encoded = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded image could not be opened. Use a JPG or PNG image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        raise ValueError("No face was detected. Upload a clear, front-facing image.")

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    margin = int(0.15 * max(width, height))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(gray.shape[1], x + width + margin)
    y1 = min(gray.shape[0], y + height + margin)
    face = cv2.resize(
        gray[y0:y1, x0:x1], (128, 128), interpolation=cv2.INTER_AREA
    )
    vector = hog(
        face,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    if len(vector) != len(feature_columns):
        raise ValueError(
            "The uploaded image produced a different number of HOG features than the trained model expects."
        )
    return pd.DataFrame([vector], columns=feature_columns)


def load_uploaded_audio(file_bytes):
    try:
        sample_rate, raw_audio = wavfile.read(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError("The voice upload must be a readable WAV file.") from exc

    if np.issubdtype(raw_audio.dtype, np.integer):
        info = np.iinfo(raw_audio.dtype)
        signal = raw_audio.astype(np.float64) / max(abs(info.min), info.max)
    else:
        signal = raw_audio.astype(np.float64)

    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = np.nan_to_num(signal)

    if sample_rate != TARGET_SR:
        common = gcd(int(sample_rate), TARGET_SR)
        signal = resample_poly(
            signal, TARGET_SR // common, int(sample_rate) // common
        )

    peak = np.max(np.abs(signal)) if len(signal) else 0
    if peak > 1:
        signal = signal / peak

    signal = signal.astype(np.float32)
    duration = len(signal) / TARGET_SR if len(signal) else 0
    rms = float(np.sqrt(np.mean(signal ** 2))) if len(signal) else 0.0
    if duration < 0.3:
        raise ValueError("The voice recording is too short. Use a recording longer than 0.3 seconds.")
    if rms < 1e-5:
        raise ValueError("The uploaded voice recording is silent or nearly silent.")
    return signal


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_filterbank(sample_rate, n_fft=512, n_mels=26):
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for index in range(1, n_mels + 1):
        left, center, right = bins[index - 1:index + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, filters.shape[1])):
            filters[index - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, filters.shape[1])):
            filters[index - 1, k] = (right - k) / (right - center)
    return filters


MEL_FILTERS = mel_filterbank(TARGET_SR)


def extract_audio_features(signal):
    _, _, spectrum = stft(
        signal, fs=TARGET_SR, nperseg=512, noverlap=352,
        boundary=None, padded=False
    )
    power = np.abs(spectrum) ** 2
    mel_power = MEL_FILTERS @ power
    log_mel = np.log(mel_power + 1e-10)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:13]

    frequencies = np.linspace(0, TARGET_SR / 2, power.shape[0])
    cumulative = np.cumsum(power, axis=0)
    totals = cumulative[-1, :] + 1e-12
    rolloff_indices = np.argmax(
        cumulative >= 0.85 * totals[None, :], axis=0
    )
    rolloff = frequencies[rolloff_indices]
    frame_energy = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=0))

    features = {}
    for index in range(13):
        features[f"mfcc_{index + 1:02d}_mean"] = float(np.mean(mfcc[index]))
        features[f"mfcc_{index + 1:02d}_std"] = float(np.std(mfcc[index]))
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))
    features["rms_energy_mean"] = float(np.mean(frame_energy))
    features["rms_energy_std"] = float(np.std(frame_energy))
    features["duration_seconds"] = float(len(signal) / TARGET_SR)
    return features


def predict_identity(model, features):
    prediction = str(model.predict(features)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)[0]))
    return prediction, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE, name="", error=None, result=None)

    name = request.form.get("name", "").strip()
    claimed_id = PARTICIPANTS.get(name.lower())
    if claimed_id is None:
        return render_template_string(
            PAGE, name=name, error="Name not recognised. Enter Ella, King, Nicole or Sarah.",
            result=None
        )

    face_file = request.files.get("face")
    voice_file = request.files.get("voice")
    if face_file is None or not face_file.filename:
        return render_template_string(
            PAGE, name=name, error="A facial image must be uploaded.", result=None
        )
    if voice_file is None or not voice_file.filename:
        return render_template_string(
            PAGE, name=name, error="A WAV voice sample must be uploaded.", result=None
        )

    try:
        face_vector = extract_face_features(
            face_file.read(), resources["detector"], resources["face_columns"]
        )
        audio_signal = load_uploaded_audio(voice_file.read())
        audio_values = extract_audio_features(audio_signal)
        audio_vector = pd.DataFrame(
            [[audio_values[column] for column in resources["audio_columns"]]],
            columns=resources["audio_columns"],
        )

        face_prediction, face_confidence = predict_identity(
            resources["face_model"], face_vector
        )
        voice_prediction, voice_confidence = predict_identity(
            resources["voice_model"], audio_vector
        )

        face_known = face_confidence is not None and face_confidence >= CONFIDENCE_THRESHOLD
        voice_known = voice_confidence is not None and voice_confidence >= CONFIDENCE_THRESHOLD
        face_ok = face_known and face_prediction == claimed_id
        voice_ok = voice_known and voice_prediction == claimed_id
        authorized = face_ok and voice_ok
        category = None
        top_scores = None

        if authorized:
            if resources["product_demo_rows"].empty:
                raise ValueError("No product rows are available for the demonstration prediction.")
            product_row = resources["product_demo_rows"].iloc[[0]]
            category = str(resources["product_model"].predict(product_row)[0])
            if hasattr(resources["product_model"], "predict_proba"):
                probabilities = resources["product_model"].predict_proba(product_row)[0]
                ranking = sorted(
                    zip(resources["product_model"].classes_, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
                top_scores = ", ".join(
                    f"{label}: {score:.1%}" for label, score in ranking
                )

        result = {
            "claimed_name": DISPLAY_NAMES[claimed_id],
            "face_name": DISPLAY_NAMES.get(face_prediction, face_prediction) if face_known else "Unknown",
            "voice_name": DISPLAY_NAMES.get(voice_prediction, voice_prediction) if voice_known else "Unknown",
            "face_confidence": f"{face_confidence:.1%}" if face_confidence is not None else "Unavailable",
            "voice_confidence": f"{voice_confidence:.1%}" if voice_confidence is not None else "Unavailable",
            "face_ok": face_ok,
            "voice_ok": voice_ok,
            "authorized": authorized,
            "category": category,
            "top_scores": top_scores,
        }
        return render_template_string(PAGE, name=name, error=None, result=result)

    except Exception as exc:
        return render_template_string(PAGE, name=name, error=str(exc), result=None)


if __name__ == "__main__":
    resources = load_resources()
    print("Open http://127.0.0.1:5000 in a browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
