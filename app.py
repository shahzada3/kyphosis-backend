from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)

# Configure CORS to allow requests from your Vercel frontend
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Update this with your Vercel domain after deployment
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Load the trained model
try:
    with open("kyphosis_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({
        'message': 'Kyphosis Prediction API',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST - Make predictions',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        
        # Extract features
        age = float(data['age'])
        num_vertebrae = float(data['numVertebrae'])
        start_vertebra = float(data['startVertebra'])
        
        # Create feature array
        features = np.array([[age, num_vertebrae, start_vertebra]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Return prediction result
        return jsonify({
            'prediction': int(prediction),
            'probability': {
                'absent': float(probability[0]),
                'present': float(probability[1])
            },
            'message': 'Prediction successful'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)