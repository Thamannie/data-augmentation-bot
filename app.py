from flask import Flask, request, jsonify
from functools import wraps
import os
import pandas as pd
from ml_processor import MLProcessor  # Import ML Processor

app = Flask(__name__)

# Configurations
app.config.update({
    'API_KEY': os.getenv('API_KEY', 'default-secret-key'),
    'DATABASE_URL': os.getenv('DATABASE_URL', 'postgresql://postgres:example@db:5432/chatbot')
})

# Initialize ML Processor
ml_processor = MLProcessor()

# ----- Authentication -----
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get('X-API-KEY') != app.config['API_KEY']:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

# ----- Botpress Webhook -----
@app.route('/webhook', methods=['POST'])
@require_api_key
def handle_webhook():
    """Handle chatbot webhook requests from Botpress"""
    data = request.json
    try:
        user_input = data['payload']['text']
        user_id = data['user']['id']

        # Get ML prediction
        prediction = ml_processor.predict([user_input])[0]

        return jsonify({
            "responses": [{"type": "text", "text": f"ML Prediction: {prediction}"}]
        })
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400

# ----- ML Prediction API -----
@app.route('/ml/predict', methods=['POST'])
@require_api_key
def ml_predict():
    """Process and return ML predictions"""
    data = request.json
    try:
        if 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400

        prediction = ml_processor.predict([data['text']])[0]
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----- ML Training API -----
@app.route('/ml/train', methods=['POST'])
@require_api_key
def train_model():
    """Train ML model using dataset"""
    try:
        ml_processor.train('/data/training_data.csv')
        return jsonify({'status': 'Training successful'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----- Start Flask App -----
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)