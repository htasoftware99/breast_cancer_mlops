import os
import pickle
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore
from config.paths_config import MODEL_OUTPUT_PATH, PROCESSED_DIR
from sklearn.preprocessing import StandardScaler
from alibi_detect.cd import KSDrift

logger = get_logger(__name__)

app = Flask(__name__, template_folder="templates")

# Feature list after preprocessing (correlation threshold=0.92, 23 features remaining)
FEATURES = [
    "texture_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "texture_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "texture_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst",
]

# Load model
try:
    model = joblib.load(MODEL_OUTPUT_PATH)
    logger.info(f"Model loaded from {MODEL_OUTPUT_PATH}")
except Exception as e:
    raise CustomException("Failed to load model", e)

# Load scaler (used for model prediction only)
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.pkl")
try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    raise CustomException("Failed to load scaler", e)

# ── Drift Detection Setup ─────────────────────────────────────────
# A separate scaler is fit on Redis reference data.
# Both the reference data and incoming inputs are scaled with this
# same drift_scaler before being passed to KSDrift.
# This matches the approach from the reference implementation.
drift_scaler = StandardScaler()
ksd          = None

def init_drift_detector():
    """
    Pulls all training records from Redis, fits a dedicated scaler,
    and initializes the KSDrift detector with the scaled reference data.
    If Redis is unavailable, drift detection is silently disabled.
    """
    global ksd, drift_scaler
    try:
        feature_store = RedisFeatureStore(host="localhost", port=6379)
        entity_ids    = feature_store.get_all_entity_ids()

        if not entity_ids:
            logger.warning("No records found in Redis. Drift detection disabled.")
            return

        all_features = feature_store.get_batch_features(entity_ids)

        ref_df = pd.DataFrame.from_dict(all_features, orient="index")
        ref_df = ref_df[[f for f in FEATURES if f in ref_df.columns]]

        # Fit drift_scaler on Redis reference data and transform
        drift_scaler.fit(ref_df)
        ref_scaled = drift_scaler.transform(ref_df)

        ksd = KSDrift(x_ref=ref_scaled, p_val=0.05)
        logger.info(f"KSDrift detector initialized. Reference shape: {ref_scaled.shape}")

    except Exception as e:
        logger.warning(f"Drift detector could not be initialized (Redis unavailable?): {e}")
        ksd = None

init_drift_detector()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction     = None
    probability    = None
    drift_detected = False
    error          = None

    if request.method == "POST":
        try:
            # Parse form inputs
            values   = [float(request.form[f]) for f in FEATURES]
            input_df = pd.DataFrame([values], columns=FEATURES)

            # Drift detection — scale with drift_scaler, then run KSDrift
            if ksd is not None:
                input_drift_scaled = drift_scaler.transform(input_df)
                drift_response     = ksd.predict(input_drift_scaled)
                is_drift           = drift_response.get("data", {}).get("is_drift", None)

                if is_drift == 1:
                    drift_detected = True
                    logger.warning("Data drift detected!")
                else:
                    logger.info("No drift detected.")

            # Prediction — scale with model scaler
            input_model_scaled  = scaler.transform(input_df)
            input_model_scaled_df = pd.DataFrame(input_model_scaled, columns=FEATURES)
            prediction  = int(model.predict(input_model_scaled_df)[0])
            probability = round(float(model.predict_proba(input_model_scaled_df)[0][prediction]) * 100, 2)

            logger.info(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'} ({probability}%) | Drift: {drift_detected}")

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            error = "Please fill in all fields correctly."

    return render_template(
        "index.html",
        features       = FEATURES,
        prediction     = prediction,
        probability    = probability,
        drift_detected = drift_detected,
        error          = error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)