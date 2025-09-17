# Used Car Price Regression

A comprehensive machine learning project for predicting used car prices using advanced regression techniques and ensemble methods.

## 📋 Project Overview

This project implements an end-to-end machine learning solution to predict used car prices using the Kaggle Playground Series S4E9 dataset. The solution includes exploratory data analysis, feature engineering, model training, validation, and deployment through a Streamlit web application.

## 🏆 Results

- **Best Model**: LightGBM Regressor
- **10-Fold CV RMSE**: 72,957
- **Performance**: Achieved top-tier accuracy with comprehensive feature engineering

## 📁 Project Structure

```
├── README.md                                           # Project documentation
├── requirements.txt                                    # Python dependencies
├── eda_and_ml_modeling.ipynb                          # Main analysis notebook
├── results.json                                       # Model performance results
├── stacking_model_results.json                       # Stacking model results
├── submission.csv                                     # Kaggle submission file
├── 12P-Presentation-Used-Car-Regression-Sahussawud-Kh.pdf  # Project presentation
├── Sahussawud-K-used-car-price-regression-solution.rar     # Complete solution archive
├── deployment_app/                                    # Streamlit deployment
│   ├── streamlit_app.py                              # Web app interface
│   ├── preprocess.py                                 # Feature engineering functions
│   └── models/                                       # Trained model files
└── playground-series-s4e9/                          # Dataset
    ├── train.csv                                     # Training data (188,533 records)
    ├── test.csv                                      # Test data (125,690 records)
    └── sample_submission.csv                         # Submission format
```

## 🔍 Dataset Overview

- **Training Data**: 188,533 used car records
- **Test Data**: 125,690 records for prediction
- **Features**: 13 columns including brand, model, year, mileage, fuel type, engine specs, colors, accident history, and clean title status
- **Target**: Car price (highly right-skewed distribution)

## 🛠️ Feature Engineering

### Key Features Created:
- **Car Age**: Derived from model year
- **Mileage per Year**: Annual usage calculation
- **Brand Categories**: Premium, Supercar, Regular classifications
- **Engine Specifications**: Extracted horsepower, engine size, and cylinder count
- **Cross Features**: Combinations of categorical variables
- **Color Normalization**: Standardized exterior and interior colors
- **Binary Flags**: Accident history, clean title status

### Data Preprocessing:
- Missing value imputation for fuel type, accident history, and clean title
- Engine specification extraction using regex patterns
- Transmission type normalization
- Log transformation for price target variable

## 🤖 Models Implemented

### Base Models (10-Fold Cross Validation Results):
1. **LightGBM Regressor**: 72,957 RMSE ⭐ *Best Performance*
2. **XGBoost Regressor**: 73,075 RMSE
3. **CatBoost Regressor**: 73,167 RMSE
4. **Random Forest Regressor**: 73,423 RMSE

### Advanced Techniques:
- **Stacking Model**: Two-stage approach with outlier detection
- **Log Target Transformation**: Improved handling of skewed price distribution
- **Cross-Feature Engineering**: Enhanced model performance with feature combinations

## 📊 Key Insights

- **Accident Impact**: Significantly reduces resale value, especially for luxury/sports cars
- **Mileage Depreciation**: Clear correlation between mileage and price reduction
- **Brand Premium**: Luxury and supercar brands command higher prices
- **Age Factor**: Newer cars maintain better value retention
- **Engine Power**: Higher horsepower correlates with increased value

## 🚀 Deployment

### Streamlit Web Application
The project includes a user-friendly web interface for real-time price predictions:

```bash
cd deployment_app
streamlit run streamlit_app.py
```

**Features:**
- Interactive form for car specifications
- Real-time price prediction
- User-friendly dropdowns for all input features
- Integrated preprocessing pipeline

## 📦 Installation & Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook eda_and_ml_modeling.ipynb
   ```

4. **Launch the web app:**
   ```bash
   cd deployment_app
   streamlit run streamlit_app.py
   ```

## 📋 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- matplotlib
- seaborn
- streamlit
- joblib

## 📈 Model Performance

The LightGBM model achieved the best performance with:
- **Stable Cross-Validation**: Consistent results across all folds
- **Feature Importance**: Effective utilization of engineered features
- **Generalization**: Strong performance on holdout data

## 🎯 Usage

1. **Training**: Run the complete notebook for model training and evaluation
2. **Prediction**: Use the saved model files for new predictions
3. **Web Interface**: Deploy the Streamlit app for interactive predictions
4. **Customization**: Modify hyperparameters and features as needed

## 📝 Notes

- Models are saved in both pickle and native formats for compatibility
- Cross-validation ensures robust performance estimation
- Feature engineering functions are modularized for reusability
- Complete preprocessing pipeline preserves data consistency

## 👤 Author

**Sahussawud Kh**
- Comprehensive ML solution with advanced feature engineering
- End-to-end implementation from EDA to deployment
- Optimized for both accuracy and interpretability