from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

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
        # Expect JSON: {"input": [[feature_vector]]} or {"input": image_array}
        data = request.get_json(force=True)
        input_data = np.array(data["input"])  # shape must match model input
        prediction = model.predict(input_data)
        pred_index = np.argmax(prediction, axis=1)[0]  # index of highest probability
        pred_label = labels[pred_index]
        confidence = float(np.max(prediction))
        return jsonify({"prediction": pred_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
