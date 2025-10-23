from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Tomato Disease Classifier API is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        img_b64 = data["input"][0]  # Expect list with single base64 image
        img_bytes = base64.b64decode(img_b64)

        # Open image and preprocess
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))  # adjust size to your model
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

        # Predict
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction, axis=1)[0]
        pred_label = labels[pred_index]
        confidence = float(np.max(prediction))

        return jsonify({"prediction": pred_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
