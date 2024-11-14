# 🎯 Salary Prediction API

Predict graduate salaries with high accuracy using our advanced ensemble ML model combining CatBoost and XGBoost! Perfect for HR professionals, career counselors, and educational institutions.

## 🌟 Key Features

- **Powerful Ensemble Model**: Combines CatBoost (60%) and XGBoost (40%) for optimal predictions
- **High Accuracy**: 98.21% variance explained (R² Score)
- **Comprehensive Feature Set**: Processes 20+ features including academics, skills, and personality traits
- **Production-Ready API**: Built with Flask, ready for deployment
- **Heroku Optimized**: Specially configured for Heroku's constraints

## 📊 Model Architecture

### Input Features

#### 📚 Academic Metrics
- 10th & 12th Percentages
- College GPA
- Graduation Year
- Board Types (10th & 12th)
- Degree & Specialization

#### 🎯 Test Scores
- English (180-875)
- Logical (195-795)
- Quantitative (120-900)

#### 💡 Domain Knowledge
| Domain | Score Range |
|--------|------------|
| Computer Programming | -1 to 804 |
| Electronics & Semiconductor | -1 to 612 |
| Computer Science | -1 to 715 |
| Mechanical Engineering | -1 to 623 |
| Electrical Engineering | -1 to 660 |
| Telecom Engineering | -1 to 548 |
| Civil Engineering | -1 to 500 |

#### 🧠 Personality Traits
- Conscientiousness (-3.89 to 1.99)
- Agreeableness (-5.78 to 1.90)
- Extraversion (-4.60 to 2.16)
- Neuroticism (-2.64 to 3.35)
- Openness (-7.37 to 1.63)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Sri-Rahul/salary-prediction-backend

# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

# Start API server
python app.py
```

## 🔌 API Endpoints

### Check Status
```bash
GET /status
```

### Get Prediction
```bash
POST /predict
Content-Type: application/json

{
    "Gender": "M",
    "10percentage": 85.0,
    "12percentage": 89.0,
    "collegeGPA": 8.5,
    ...
}
```

## 📈 Performance Metrics

```python
{
    'MSE': 807_102_031.60,
    'RMSE': 28_409.54,
    'MAE': 21_190.95,
    'R² Score': 0.9821  # 98.21% variance explained
}
```

## 🛠️ Technical Stack

### Dependencies
```
numpy==1.21.0
pandas==1.3.0
scikit-learn==1.5.2
catboost==1.0.0
xgboost==1.5.0
Flask==2.0.1
flask-cors==3.0.10
gunicorn==20.1.0
joblib==1.2.0
```

### Model Pipeline
1. Feature validation
2. Label encoding
3. Ensemble prediction
4. Performance metrics calculation

## ⚠️ Deployment Notes

- Optimized for Heroku's 500MB slug size limit
- Uses specific dependency versions for compatibility
- Models are serialized efficiently for quick loading

## 🔍 Monitoring & Error Handling

- Comprehensive logging system
- Feature validation
- Graceful error responses
- Performance tracking
- Model status monitoring

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
Made with ❤️ by Rahul
