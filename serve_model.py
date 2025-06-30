from flask import Flask, request, jsonify 
import tensorflow as tf
import numpy as np
import os
from prometheus_client import start_http_server, Counter

# Prometheus metric
REQUEST_COUNT = Counter('request_count', 'Total request count')

app = Flask(__name__)

# Lokasi model hasil push dari pipeline
PUSHED_MODEL_DIR = os.path.join(os.getcwd(), 'dzikrimaulana87-pipeline', 'churn_pipeline', 'Pusher', 'pushed_model')

# Cari folder versi terbaru
all_versions = [d for d in os.listdir(PUSHED_MODEL_DIR) if d.isdigit()]
latest_version = max(all_versions, key=int)
MODEL_DIR = os.path.join(PUSHED_MODEL_DIR, latest_version)

# Load model
model = tf.keras.models.load_model(MODEL_DIR)

@app.route('/')
def home():
    return """
    <html>
    <body style="font-family: monospace; padding: 20px;">
    <h2>✅ Churn Prediction Model is Running!</h2>
    
    <h3>📌 Available Endpoints:</h3>
    
    <p><strong>1. [POST] /predict</strong><br>
    &nbsp;&nbsp;&nbsp;- Description : Predict churn probability from input features<br>
    &nbsp;&nbsp;&nbsp;- Request JSON:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"feature1": [value], "feature2": [value], ..., "featureN": [value]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}<br>
    &nbsp;&nbsp;&nbsp;- Response JSON:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"prediction": [[value]]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}</p>
    
    <p><strong>2. [GET] /metrics</strong><br>
    &nbsp;&nbsp;&nbsp;- Description : Prometheus-compatible metrics endpoint<br>
    &nbsp;&nbsp;&nbsp;- Output : Counter for request count and other metrics</p>
    
    <h3>📎 Example curl usage:</h3>
    <code>curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{ "gender_xf": [[0.0, 1.0]], ... }'</code>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNT.inc()
    try:
        input_json = request.get_json(force=True)
        if not isinstance(input_json, dict):
            return jsonify({'error': 'Invalid input format, expected JSON object'}), 400

        # Convert each value to Tensor, reshape if needed
        input_data = {}
        for k, v in input_json.items():
            tensor = tf.convert_to_tensor(v)
            # If shape is (1, X), and model expects (1, 1, X), expand dims
            if len(tensor.shape) == 2 and tensor.shape[0] == 1:
                expected_input_shape = None
                if isinstance(model.input, dict) and k in model.input:
                    expected_input_shape = model.input[k].shape
                elif isinstance(model.input, (list, tuple)):
                    try:
                        idx = list(input_json.keys()).index(k)
                        expected_input_shape = model.input[idx].shape
                    except:
                        pass

                if len(expected_input_shape) == 3 and expected_input_shape[1] == 1:
                    tensor = tf.expand_dims(tensor, axis=1)  # (1, X) → (1, 1, X)
            input_data[k] = tensor

        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    # Start Prometheus metrics server di port 8000
    start_http_server(8000)  # Ini expose /metrics
    app.run(host='0.0.0.0', port=8080)