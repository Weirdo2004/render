from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import sys # Import sys for flushing output

app = Flask(__name__)

# --- Load model ---
try:
    MODEL_PATH = "model.h5"
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.", flush=True)
except Exception as e:
    print(f"Error loading model: {e}", flush=True)
    model = None # Set model to None to handle errors

# --- Load labels ---
try:
    with open("labels.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Labels loaded: {labels}", flush=True)
except Exception as e:
    print(f"Error loading labels.txt: {e}", flush=True)
    labels = []

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Tomato Disease Classifier API is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    # --- ADDED LOGS ---
    print("\n--- Received prediction request ---", flush=True)
    
    if model is None or not labels:
        print("Error: Model or labels are not loaded.", flush=True)
        return jsonify({"error": "Server configuration error: Model or labels missing."}), 500
        
    try:
        data = request.get_json(force=True)
        img_b64 = data["input"][0]  # Expect list with single base64 image
        img_bytes = base64.b64decode(img_b64)
        
        print("Image decoded from base64.", flush=True)

        # Open image and preprocess
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # *** IMPORTANT ***
        # This size MUST match your model's input. 
        # (224, 224) is common, but check your model.
        target_size = (224, 224) 
        img = img.resize(target_size) 
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
        
        print(f"Image preprocessed. Array shape: {img_array.shape}", flush=True)

        # Predict
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction, axis=1)[0]
        pred_label = labels[pred_index]
        confidence = float(np.max(prediction))

        print(f"Prediction: {pred_label}, Confidence: {confidence}", flush=True)
        # --- END LOGS ---

        return jsonify({"prediction": pred_label, "confidence": confidence})
        
    except Exception as e:
        # --- ADDED LOG ---
        print(f"!!! ERROR processing request: {str(e)}", flush=True)
        # --- END LOG ---
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # This port is specified by Render.
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host="0.0.0.0", port=port)