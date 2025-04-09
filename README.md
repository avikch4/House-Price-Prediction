# 🏠 House Price Prediction - Advanced ML Pipeline

This project provides a complete end-to-end machine learning pipeline for predicting house prices using various regression models. It includes:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing with Pipelines
- Model Building & Evaluation (7 models)
- Hyperparameter Tuning (Grid Search)
- Feature Importance Analysis (using `coef_` or `feature_importances_`)
- Outputs for Power BI / Tableau visualization
- Saved plots and metrics for insights

---

## 📁 Files Generated

After running the full pipeline, the following files will be created:

### 📊 Visualizations
- `price_distribution.png` – Distribution of sale prices
- `log_price_distribution.png` – Log-transformed sale price distribution
- `top_correlations.png` – Top features correlated with price
- `scatterplots.png` – Scatter plots for key features
- `missing_values.png` – Top features with missing data
- `categorical_features.png` – Median price by top categorical features
- `model_comparison.png` – RMSE and R² for all models
- `prediction_plot_<Model>.png` – Actual vs predicted for each model
- `feature_importance.png` / `feature_coefficients.png` – Most important predictors

### 📄 CSV Outputs (for dashboard use)
- `prediction_results.csv` – Actual vs predicted values with errors
- `neighborhood_analysis.csv` – Summary stats by neighborhood
- `error_analysis.csv` – Categorized prediction errors

---

## 🧪 Models Used

- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost
- LightGBM

Each model is evaluated using:
- RMSE
- MAE
- R² Score

---

## 📦 Dependencies

```bash
conda env create -f house-price-prediction.yml
conda activate house-price-prediction
