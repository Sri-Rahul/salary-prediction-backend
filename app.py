import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from salary_prediction import EnhancedSalaryPredictor
import pandas as pd
import traceback
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize predictor and model paths
predictor = EnhancedSalaryPredictor()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
predictor.load_models_separately(MODEL_DIR)

@app.route('/status', methods=['GET'])
def get_model_status():
    try:
        status = predictor.get_model_status()
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = request.get_json()
        if not input_data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400

        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Validate features
        predictor.validate_features(df)
        
        # Make prediction
        prediction = predictor.predict(df)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction[0])
        })
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Get input data
        data = request.get_json()
        if not data or 'X' not in data or 'y' not in data:
            return jsonify({'success': False, 'error': 'Missing X or y data'}), 400

        # Convert to DataFrames
        X = pd.DataFrame(data['X'])
        y = np.array(data['y'])
        
        # Validate features
        predictor.validate_features(X)
        
        # Evaluate
        metrics = predictor.evaluate(X, y)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/load_models', methods=['POST'])
def load_models():
    try:
        success = predictor.load_models_separately(MODEL_DIR)
        if success:
            return jsonify({'success': True, 'message': 'Models loaded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to load models'}), 500
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

def main():
    # Initialize predictor
    predictor = EnhancedSalaryPredictor()

    # Load your training data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'path_to_your_training_data.csv')
    data = pd.read_csv(DATA_PATH)

    # Separate features and target variable
    X = data.drop('Salary', axis=1)
    y = data['Salary']

    # Fit the predictor
    predictor.fit(X, y)
    logger.info("Models trained successfully.")

    # Save the models
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    save_paths = predictor.save_models_separately(MODEL_DIR)
    if save_paths:
        logger.info(f"Models saved successfully:")
        for model_type, path in save_paths.items():
            logger.info(f"{model_type} model saved at: {path}")
    else:
        logger.error("Error saving models.")

if __name__ == '__main__':
    # Load models on startup
    try:
        predictor.load_models_separately(MODEL_DIR)
        logger.info("Models loaded successfully on startup")
    except Exception as e:
        logger.error(f"Error loading models on startup: {str(e)}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
    main()