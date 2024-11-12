from salary_prediction import EnhancedSalaryPredictor
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize predictor
    predictor = EnhancedSalaryPredictor()
    
    # Load training data
    training_data_url = 'https://raw.githubusercontent.com/Sri-Rahul/salary-prediction-backend/main/graduate_engineering_salary.csv'
    logger.info("Loading training data...")
    df = pd.read_csv(training_data_url)
    
    # Split features and target
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    
    # Train model
    logger.info("Training model...")
    predictor.fit(X, y)
    
    # Evaluate model
    performance = predictor.evaluate(X, y)
    logger.info(f"Training completed with performance: {performance}")
    
    # Define base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save models using the new method
    try:
        save_paths = predictor.save_models_separately(models_dir)
        logger.info("Models saved successfully:")
        for model_type, path in save_paths.items():
            logger.info(f"- {model_type}: {path}")
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        raise

if __name__ == "__main__":
    main()