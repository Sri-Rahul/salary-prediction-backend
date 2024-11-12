# 🎓 Salary Prediction API

A machine learning-powered API that predicts graduate salaries using an ensemble of CatBoost and XGBoost models.

## 🌟 Features

### 📊 Model Architecture
- **Ensemble Model**: Weighted combination of:
  - CatBoost Regressor (60% weight)
  - XGBoost Regressor (40% weight)

### 🔧 Feature Engineering

#### Categorical Features
- Gender
- 10th Board
- 12th Board
- Degree
- Specialization
- College Tier
- College City Tier
- College State
- Domain

#### Numerical Features
- 10th Percentage
- 12th Percentage
- College GPA
- Graduation Year
- Test Scores:
  - English (180-875)
  - Logical (195-795)
  - Quant (120-900)
- Domain Knowledge:
  - Computer Programming (-1 to 804)
  - Electronics & Semiconductor (-1 to 612)
  - Computer Science (-1 to 715)
  - Mechanical Engineering (-1 to 623)
  - Electrical Engineering (-1 to 660)
  - Telecom Engineering (-1 to 548)
  - Civil Engineering (-1 to 500)

#### Personality Traits
- Conscientiousness (-3.8933 to 1.9953)
- Agreeableness (-5.7816 to 1.9048)
- Extraversion (-4.6009 to 2.1617)
- Neuroticism (-2.643 to 3.3525)
- Openness to Experience (-7.3757 to 1.6302)

## 🛠️ Technical Implementation

### API Endpoints
- `GET /status`: Check model status
- `POST /predict`: Get salary predictions

### Model Training Pipeline
## Prediction Pipeline
-Feature validation
-Label encoding
-Model ensemble prediction
-Performance metrics (MSE, RMSE, MAE, R²)

### 📦 Dependencies
numpy==1.21.0
pandas==1.3.0
scikit-learn==1.5.2
catboost==1.0.0
xgboost==1.5.0
Flask==2.0.1
flask-cors==3.0.10
gunicorn==20.1.0
joblib==1.2.0

### ⚠️ Heroku Deployment Note
This project uses older versions of some dependencies due to Heroku's 500MB slug size limit. The models are optimized for deployment while maintaining prediction accuracy.

### 📈 Performance Metrics
The model is evaluated using:

-Mean Squared Error (MSE)
-Root Mean Squared Error (RMSE)
-Mean Absolute Error (MAE)
-R² Score

### 🔄 Model Lifecycle
-Data preprocessing
-Model training
-Model serialization
-API deployment
-Real-time predictions

#### 🚀 Quick Start

# Clone repository
git clone <repository-url>

# Install dependencies
pip install -r [requirements.txt](http://_vscodecontentref_/0)

# Train models
python [train_models.py](http://_vscodecontentref_/1)

# Start API server
python [app.py](http://_vscodecontentref_/2)

### 📡 API Usage

# Example prediction request
```bash
POST /predict
{
    "Gender": "M",
    "10percentage": 85.0,
    "12percentage": 89.0,
    ...
}
```
#### 🛡️ Error Handling
-Robust feature validation
-Comprehensive error logging
-Graceful API error responses

#### 📊 Model Storage
-Models stored separately for efficient loading
-Metadata saved with label encoders
-Feature order preservation

#### 🔍 Monitoring
-Detailed logging system
-Model status tracking
-Performance metrics monitoring

#### 📈 Performance Metrics
The model achieves excellent performance with the following metrics:
```python
{
    'MSE': 807,102,031.60
    'RMSE': 28,409.54
    'MAE': 21,190.95
    'R² Score': 0.9821 (98.21% variance explained)
}
```