from flask import Flask, render_template, request, session, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key-for-local-use")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_auto_model.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.json")

# Load the model saved from your notebook.
model = joblib.load(MODEL_PATH)

# Use the feature order saved with the model where available.
FEATURES = list(getattr(model, "feature_names_in_", []))
if not FEATURES:
    # Fallback for models saved without DataFrame feature names.
    FEATURES = [
        "symboling", "wheel-base", "length", "width", "height",
        "curb-weight", "num-of-cylinders", "engine-size",
        "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
        "highway-mpg"
    ]

LABELS = {
    "symboling": ("Symboling", "Insurance risk rating; use the same encoding as training data."),
    "wheel-base": ("Wheel Base", "Wheelbase measurement in the dataset's units."),
    "length": ("Length", "Vehicle length in the dataset's units."),
    "width": ("Width", "Vehicle width in the dataset's units."),
    "height": ("Height", "Vehicle height in the dataset's units."),
    "curb-weight": ("Curb Weight", "Vehicle curb weight in the dataset's units."),
    "num-of-cylinders": ("Number of Cylinders (encoded)", "Enter the numeric encoding used during model training."),
    "engine-size": ("Engine Size", "Engine size in the dataset's units."),
    "compression-ratio": ("Compression Ratio", "Engine compression ratio."),
    "horsepower": ("Horsepower", "Horsepower value."),
    "peak-rpm": ("Peak RPM", "Engine peak revolutions per minute."),
    "city-mpg": ("City MPG", "City fuel economy in the dataset's units."),
    "highway-mpg": ("Highway MPG", "Highway fuel economy in the dataset's units.")
}

def load_history():
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history[:100], f, indent=2)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {}
    history = load_history()

    if request.method == "POST":
        try:
            row = []
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                if raw == "":
                    raise ValueError(f"Please enter {LABELS.get(feature, (feature, ''))[0]}.")
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError(f"Please enter a valid number for {LABELS.get(feature, (feature, ''))[0]}.")
                values[feature] = value
                row.append(value)

            input_df = pd.DataFrame([row], columns=FEATURES)
            prediction = float(model.predict(input_df)[0])
            if not np.isfinite(prediction):
                raise ValueError("The model returned an invalid prediction.")

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "prediction": round(prediction, 2),
                "inputs": values.copy()
            }
            history.insert(0, record)
            save_history(history)
            history = load_history()

        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Check that your model and input feature columns match."

    fields = [
        {"name": feature,
         "label": LABELS.get(feature, (feature.replace("_", " ").replace("-", " ").title(), ""))[0],
         "help": LABELS.get(feature, ("", "Enter a numerical value."))[1],
         "value": values.get(feature, "")}
        for feature in FEATURES
    ]

    # Feature importance is available for DecisionTreeRegressor.
    importance = []
    if hasattr(model, "feature_importances_"):
        pairs = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
        importance = [
            {"label": LABELS.get(name, (name, ""))[0], "value": round(float(score) * 100, 2)}
            for name, score in pairs[:8]
        ]

    return render_template(
        "index.html",
        fields=fields,
        prediction=prediction,
        error=error,
        history=history[:8],
        importance=importance,
        feature_count=len(FEATURES),
        model_name=type(model).__name__
    )

@app.route("/clear-history", methods=["POST"])
def clear_history():
    save_history([])
    flash("Prediction history cleared.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    # Debug mode is convenient locally; disable it for public deployment.
    app.run(debug=True)
