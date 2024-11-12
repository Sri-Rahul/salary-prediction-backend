# salary_prediction.py
import logging
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import traceback

class EnhancedSalaryPredictor:
    def __init__(self):
        # Previous initialization code remains the same
        self.logger = logging.getLogger(__name__)
        
        # Feature definitions remain the same
        self.ignore_features = ['ID', 'CollegeID', 'CollegeCityID', '12graduation']
        self.categorical_features = [
            'Gender', '10board', '12board', 'Degree', 'Specialization',
            'CollegeTier', 'CollegeCityTier', 'CollegeState', 'Domain'
        ]
        self.numerical_features = [
            '10percentage', '12percentage', 'collegeGPA', 'GraduationYear',
            'English', 'Logical', 'Quant', 'ComputerProgramming',
            'ElectronicsAndSemicon', 'ComputerScience', 'MechanicalEngg',
            'ElectricalEngg', 'TelecomEngg', 'CivilEngg'
        ]
        self.personality_features = [
            'conscientiousness', 'agreeableness', 'extraversion', 
            'nueroticism', 'openess_to_experience'
        ]
        
        # Feature bounds remain the same
        self.feature_bounds = {
            'Domain': (-1, 0.999910408),
            'ComputerProgramming': (-1, 804),
            'ElectronicsAndSemicon': (-1, 612),
            'ComputerScience': (-1, 715),
            'MechanicalEngg': (-1, 623),
            'ElectricalEngg': (-1, 660),
            'TelecomEngg': (-1, 548),
            'CivilEngg': (-1, 500),
            'English': (180, 875),
            'Logical': (195, 795),
            'Quant': (120, 900),
            'conscientiousness': (-3.8933, 1.9953),
            'agreeableness': (-5.7816, 1.9048),
            'extraversion': (-4.6009, 2.1617),
            'nueroticism': (-2.643, 3.3525),
            'openess_to_experience': (-7.3757, 1.6302)
        }
        
        self.label_encoders = {}
        self.models = {}
        self.feature_order = None

    def validate_features(self, data):
        """Validate feature values are within acceptable ranges"""
        try:
            for feature, (min_val, max_val) in self.feature_bounds.items():
                if feature in data.columns:
                    values = data[feature]
                    if values.min() < min_val or values.max() > max_val:
                        raise ValueError(
                            f"Feature '{feature}' values must be between {min_val} and {max_val}."
                        )
            return True
        except Exception as e:
            self.logger.error(f"Feature validation error: {str(e)}")
            raise

    def preprocess_data(self, data):
        """Preprocess data by removing ignored features and encoding categoricals"""
        try:
            data = data.copy()
            
            # Remove ignored features
            data = data.drop(columns=self.ignore_features, errors='ignore')
            
            # Handle categorical features
            for feature in self.categorical_features:
                if feature in data.columns:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                        data[feature] = self.label_encoders[feature].fit_transform(data[feature])
                    else:
                        try:
                            data[feature] = self.label_encoders[feature].transform(data[feature])
                        except ValueError as e:
                            self.logger.error(f"Error transforming feature {feature}: {str(e)}")
                            raise
            
            # Store/verify feature order
            if self.feature_order is None:
                self.feature_order = [col for col in data.columns if col not in self.ignore_features]
            
            # Ensure columns match training order
            return data[self.feature_order]
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def save_models_separately(self, directory):
        os.makedirs(directory, exist_ok=True)
        save_paths = {}
        try:
            # Save metadata
            metadata = {
                'label_encoders': self.label_encoders,
                'feature_order': self.feature_order,
                'model_params': {
                    'catboost': self.models['catboost'].get_params(),
                    'xgboost': self.models['xgboost'].get_params()
                }
            }
            metadata_path = os.path.join(directory, 'metadata.joblib')
            joblib.dump(metadata, metadata_path)
            save_paths['metadata'] = metadata_path

            # Save CatBoost model
            catboost_path = os.path.join(directory, 'catboost_model.cbm')
            self.models['catboost'].save_model(catboost_path)
            self.logger.info("CatBoost model saved successfully")
            save_paths['catboost'] = catboost_path

            # Save XGBoost model in binary format using '.deprecated' extension
            xgboost_path = os.path.join(directory, 'xgboost_model.deprecated')
            self.models['xgboost'].save_model(xgboost_path)
            self.logger.info("XGBoost model saved successfully")
            save_paths['xgboost'] = xgboost_path

            self.logger.info("All models and metadata saved successfully")
            return save_paths
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            return None

    def load_models_separately(self, directory):
        try:
            # Load metadata
            metadata_path = os.path.join(directory, 'metadata.joblib')
            metadata = joblib.load(metadata_path)
            self.label_encoders = metadata['label_encoders']
            self.feature_order = metadata['feature_order']

            # Initialize models with saved parameters
            self.models['catboost'] = CatBoostRegressor()
            self.models['xgboost'] = XGBRegressor(**metadata['model_params']['xgboost'])

            # Load CatBoost model
            catboost_path = os.path.join(directory, 'catboost_model.cbm')
            self.models['catboost'].load_model(catboost_path)
            self.logger.info("CatBoost model loaded successfully")

            # Load XGBoost model from binary file with '.deprecated' extension
            xgboost_path = os.path.join(directory, 'xgboost_model.deprecated')
            self.models['xgboost'].load_model(xgboost_path)
            self.logger.info("XGBoost model loaded successfully")

            self.logger.info("All models and metadata loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}\n{traceback.format_exc()}")
            return False

    def fit(self, X, y):
        """Train the models with consistent feature ordering and error handling"""
        try:
            # Store feature order from training data
            self.feature_order = X.columns.tolist()
            self.feature_order = [f for f in self.feature_order if f not in self.ignore_features]
            
            X_processed = self.preprocess_data(X)
            
            # Initialize and train CatBoost
            self.models['catboost'] = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.1,
                verbose=False
            )
            self.models['catboost'].fit(X_processed, y)
            self.logger.info("CatBoost model trained successfully")
            
            # Initialize and train XGBoost
            self.models['xgboost'] = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.1,
                verbosity=0
            )
            self.models['xgboost'].fit(X_processed, y)
            self.logger.info("XGBoost model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}\n{traceback.format_exc()}")
            raise

    def predict(self, X):
        """Predict using models with correct feature ordering and error handling"""
        try:
            if self.feature_order is None:
                raise ValueError("Model not fitted. Call fit() before predict()")
            
            if not self.models:
                raise ValueError("No models loaded. Either train models or load pre-trained models.")
            
            # Ensure input features match training order
            X = X[self.feature_order]
            X_processed = self.preprocess_data(X)
            
            # Make predictions with both models
            catboost_pred = self.models['catboost'].predict(X_processed)
            xgboost_pred = self.models['xgboost'].predict(X_processed)
            
            # Ensemble predictions with weights
            final_predictions = 0.6 * catboost_pred + 0.4 * xgboost_pred
            
            return final_predictions
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}\n{traceback.format_exc()}")
            raise

    def evaluate(self, X, y):
        """Evaluate model performance with error handling"""
        try:
            predictions = self.predict(X)
            return {
                'mse': mean_squared_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'mae': mean_absolute_error(y, predictions),
                'r2': r2_score(y, predictions)
            }
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise

    def get_model_status(self):
        """Get the current status of model components"""
        return {
            'models_loaded': bool(self.models),
            'catboost_loaded': 'catboost' in self.models,
            'xgboost_loaded': 'xgboost' in self.models,
            'feature_order_set': self.feature_order is not None,
            'label_encoders_set': bool(self.label_encoders),
            'number_of_features': len(self.feature_order) if self.feature_order else 0
        }