from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
from pathlib import Path
import pandas as pd
import requests
import os

# === Paths and Directories ===
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATASET_PATH = ROOT_DIR / "dataset" / "house_price_data_100k.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

# === Hugging Face API Setup ===
API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
HF_TOKEN = "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"  # Use env variable in production
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# === Flask App ===
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


# === Model Utility Functions ===
def ensure_model():
    """Ensure model.pkl exists; if not, train and save it."""
    if not MODEL_PATH.exists():
        from train_model import train_and_save
        print("⚡ Model not found, training a new one...")
        train_and_save()
        print("✅ Model trained and saved.")


def load_model():
    """Load trained ML model."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def query_llm(newdata, prediction):
    """
    Query Hugging Face API to generate a short explanation
    for the predicted house price.
    """
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"Predicting house price. Input: {newdata}. Predicted price: {prediction} lakhs. Explain briefly."
            }
        ],
        "model": "accounts/fireworks/models/deepseek-r1"
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No explanation available.")
    except Exception as e:
        return f"Explanation API error: {str(e)}"


# === Ensure model is loaded at startup ===
with app.app_context():
    ensure_model()
    model = load_model()


# === Routes ===
@app.get("/")
def index():
    """Serve the frontend index page."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/options")
def options():
    """Fetch dropdown options for frontend dynamically from dataset."""
    if not DATASET_PATH.exists():
        return jsonify({"error": "Dataset not found!"}), 404
    df = pd.read_csv(DATASET_PATH)
    locations = sorted([str(x) for x in df["Location"].dropna().unique()])
    nearby_metro = sorted([str(x) for x in df["Nearby Metro"].dropna().unique()])
    return jsonify({
        "locations": locations,
        "nearby_metro": nearby_metro
    })


@app.post("/predict")
def predict():
    """Predict house price and generate an explanation."""
    data = request.get_json(force=True)
    required = ["Bedrooms", "Bathrooms", "Area", "Year Built", "Location", "Nearby Metro"]
    missing = [k for k in required if k not in data]

    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        # Prepare input row
        row = {
            "Bedrooms": float(data["Bedrooms"]),
            "Bathrooms": float(data["Bathrooms"]),
            "Area": float(data["Area"]),
            "Year Built": float(data["Year Built"]),
            "Location": str(data["Location"]),
            "Nearby Metro": str(data["Nearby Metro"]),
        }
        df = pd.DataFrame([row])

        # Make prediction
        prediction = round(float(model.predict(df)[0]), 2)

        # Get explanation from Hugging Face
        explanation = query_llm(row, prediction)

        return jsonify({
            "prediction": prediction,
            "unit": "lakhs",
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting backend server with explanation feature...")
    app.run(host="0.0.0.0", port=5000, debug=True)
