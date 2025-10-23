from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import sys
# Import the TFLite runtime
import tflite_runtime.interpreter as tflite 

app = Flask(__name__)

# --- Load TFLite model ---
try:
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensor details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get the required input size from the model
    # This is safer than hardcoding (224, 224)
    _, input_height, input_width, _ = input_details[0]['shape']
    
    print(f"Model loaded. Input size: ({input_height}, {input_width})", flush=True)

except Exception as e:
    print(f"Error loading model: {e}", flush=True)
    interpreter = None

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
    print("\n--- Received prediction request ---", flush=True)
    
    if interpreter is None or not labels:
        print("Error: Model or labels are not loaded.", flush=True)
        return jsonify({"error": "Server configuration error: Model or labels missing."}), 500
        
    try:
        data = request.get_json(force=True)
        img_b64 = data["input"][0]
        img_bytes = base64.b64decode(img_b64)
        print("Image decoded from base64.", flush=True)

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Resize to the model's required input size
        img = img.resize((input_width, input_height))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension and ensure it's float32
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        print(f"Image preprocessed. Array shape: {img_array.shape}", flush=True)

        # --- TFLite Prediction ---
        # Set the tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()

        # Get the results
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        # --- End TFLite Prediction ---

        pred_index = np.argmax(prediction)
        pred_label = labels[pred_index]
        
        # TFLite outputs probabilities (0-1), not raw logits
        confidence = float(np.max(prediction))

        print(f"Prediction: {pred_label}, Confidence: {confidence}", flush=True)

        return jsonify({"prediction": pred_label, "confidence": confidence})
        
    except Exception as e:
        print(f"!!! ERROR processing request: {str(e)}", flush=True)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)