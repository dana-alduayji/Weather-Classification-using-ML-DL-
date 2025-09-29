# Weather Classification Project

## 📋 Project Overview

This project implements and compares multiple models for classifying weather conditions based on meteorological data. Using a comprehensive dataset from Kaggle, we developed and evaluated Random Forest and Neural Network models to predict four weather types: Rainy, Cloudy, Sunny, and Snowy.

## 🏆 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Best For |
|-------|----------|-----------|--------|----------|----------|
| Random Forest (Base) | 91.14% | 91.16% | 91.14% | 91.12% | 🏆 Best Overall |
| Random Forest (GridSearch) | 91.00% | 91.00% | 91.00% | 91.00% | Optimized Parameters |
| Neural Network | 90.00% | 90.00% | 90.00% | 90.00% | Complex Patterns |

## 📊 Detailed Performance Analysis

### Random Forest (Base Model) - RECOMMENDED

```
              precision    recall  f1-score   support

      Cloudy     0.8986    0.8680    0.8830       674
       Rainy     0.9119    0.9021    0.9070       654
       Snowy     0.9401    0.9334    0.9367       706
       Sunny     0.8924    0.9439    0.9174       606

    accuracy                         0.9114      2640
   macro avg     0.9107    0.9119    0.9110      2640
weighted avg     0.9116    0.9114    0.9112      2640
```

### Random Forest (GridSearch Optimized)

```
              precision    recall  f1-score   support

      Cloudy       0.87      0.90      0.89       651
       Rainy       0.90      0.90      0.90       647
       Snowy       0.93      0.94      0.94       701
       Sunny       0.95      0.90      0.93       641

    accuracy                           0.91      2640
   macro avg       0.91      0.91      0.91      2640
weighted avg       0.91      0.91      0.91      2640
```

### Neural Network (PyTorch)

```
              precision    recall  f1-score   support

      Cloudy       0.87      0.88      0.88       651
       Rainy       0.88      0.89      0.89       647
       Snowy       0.94      0.94      0.94       701
       Sunny       0.92      0.90      0.91       641

    accuracy                           0.90      2640
   macro avg       0.90      0.90      0.90      2640
weighted avg       0.90      0.90      0.90      2640
```

## 🎯 Key Findings

### Performance by Weather Type

- **❄️ Snowy Weather**: Easiest to classify (93-94% F1-score across all models)
- **☀️ Sunny Weather**: Best precision with GridSearch RF (95%)
- **🌧️ Rainy Weather**: Most balanced with Base RF (90.7% F1-score)
- **☁️ Cloudy Weather**: Most challenging but still strong (87-89% F1-score)

### Model Insights

- **Random Forest Superiority**: Both RF variants outperform the neural network
- **Hyperparameter Tuning**: GridSearch provided more balanced class performance
- **Model Robustness**: All models achieve >90% accuracy
- **Computational Efficiency**: Base RF offers best performance without extensive tuning

## 🗃️ Dataset

- **Source**: Kaggle "Weather Type Classification" dataset
- **Samples**: 13,200 weather observations
- **Features**: 19 engineered features

### Feature Categories

**🌡️ Numerical Features:**
- Temperature, Humidity, Wind Speed
- Precipitation (%), Atmospheric Pressure
- UV Index, Visibility (km)

**📊 Categorical Features (One-hot encoded):**
- Cloud Cover: clear, cloudy, overcast, partly cloudy
- Season: Winter, Spring, Summer, Autumn
- Location: inland, mountain, coastal

## 🛠️ Technical Implementation

### Models Developed

#### 1. Random Forest Classifier

- **Base Model**: Default parameters with excellent performance
- **GridSearchCV**: Hyperparameter optimization

```python
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
```

#### 2. Neural Network (PyTorch)

```python
class WeatherClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(WeatherClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.dropout = nn.Dropout(0.3)
```

### Data Preprocessing Pipeline

1. Data Loading & Exploration
2. Feature Engineering: One-hot encoding for categorical variables
3. Label Encoding: Target variable transformation
4. Standard Scaling: Numerical feature normalization
5. Train-Validation Split: 80-20 split

## 🚀 Installation & Usage

### Prerequisites

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn
```

### Quick Start

1. Clone the repository
2. Run the Jupyter notebook `WeatherClassification.ipynb`
3. The notebook will:
   - Automatically download the dataset
   - Preprocess the data
   - Train all models
   - Generate performance comparisons

## 📈 Results Visualization

The project includes comprehensive visualizations:

- Confusion Matrices for each model
- Classification Reports with detailed metrics
- Training Progress plots for neural network
- Model Comparison charts

## 💡 Recommendations

### For Production Deployment

**🏆 Primary Choice**: Base Random Forest (91.14% accuracy)

**🎯 Use Case Specific:**
- **Highest Accuracy**: Base Random Forest
- **Best Sunny Detection**: GridSearch Random Forest
- **Research/Experimentation**: Neural Network

### Why Choose Base Random Forest?

- Highest overall accuracy (91.14%)
- Most balanced performance across all classes
- No hyperparameter tuning required
- Faster training and inference
- Excellent interpretability

## 🔮 Future Enhancements

- Expand hyperparameter search space
- Implement ensemble methods
- Add feature importance analysis
- Develop web deployment interface
- Incorporate real-time weather data streaming
- Add cross-validation for neural network

## 📝 License

This project uses the Weather Type Classification dataset from Kaggle. Please refer to the original dataset for specific licensing information.

## 👥 Author

Weather Classification Project - Comprehensive ML Comparison Study