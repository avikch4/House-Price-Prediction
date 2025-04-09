import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
def load_data(file_path="house_prices.csv"):
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

# Exploratory Data Analysis
def explore_data(df):
    # Basic stats
    print("\nBasic statistics for target variable (SalePrice):")
    print(df['SalePrice'].describe())
    
    # Target variable distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title('Distribution of House Prices')
    plt.savefig('price_distribution.png')
    
    # Check skewness
    skewness = df['SalePrice'].skew()
    print(f"\nSkewness of SalePrice: {skewness}")
    
    if abs(skewness) > 0.5:
        # Log transform to handle skewness
        plt.figure(figsize=(10, 6))
        sns.histplot(np.log1p(df['SalePrice']), kde=True)
        plt.title('Distribution of Log-Transformed House Prices')
        plt.savefig('log_price_distribution.png')
        
        print("Log transformation applied to handle skewness")
        df['SalePrice_Log'] = np.log1p(df['SalePrice'])
    else:
        df['SalePrice_Log'] = df['SalePrice']
    
    # Correlation with numerical features
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlations
    correlations = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
    print("\nTop 10 features correlated with SalePrice:")
    print(correlations.head(10))
    
    # Plot top correlations
    top_corr = correlations[1:11]  # Exclude SalePrice itself
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title('Top 10 Features Correlated with Sale Price')
    plt.tight_layout()
    plt.savefig('top_correlations.png')
    
    # Scatter plots for top correlated features
    top_features = correlations[1:6].index  # Top 5 correlated features
    
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(3, 2, i)
        sns.scatterplot(x=df[feature], y=df['SalePrice'])
        plt.title(f'SalePrice vs {feature}')
    plt.tight_layout()
    plt.savefig('scatterplots.png')
    
    # Visualize categorical features
    plt.figure(figsize=(15, 10))
    categorical_cols = df.select_dtypes(include=['object']).columns[:5]  # Select first 5 for brevity
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(3, 2, i)
        data = df.groupby(col)['SalePrice'].median().sort_values(ascending=False)
        sns.barplot(x=data.index, y=data.values)
        plt.title(f'Median SalePrice by {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('categorical_features.png')
    
    return df

# Data Preprocessing
def preprocess_data(df):
    # Drop ID column if exists
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Separate target
    if 'SalePrice_Log' in df.columns:
        y = df['SalePrice_Log']
        X = df.drop(['SalePrice', 'SalePrice_Log'], axis=1)
    else:
        y = df['SalePrice']
        X = df.drop('SalePrice', axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle missing values analysis
    missing_values = X_train.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    missing_values.head(10).plot(kind='bar')
    plt.title('Top Features with Missing Values')
    plt.tight_layout()
    plt.savefig('missing_values.png')
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

# Feature Engineering (optional)
def feature_engineering(X_train, X_test):
    # Example feature: Total Square Footage
    if 'TotalBsmtSF' in X_train.columns and '1stFlrSF' in X_train.columns and '2ndFlrSF' in X_train.columns:
        X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']
        X_test['TotalSF'] = X_test['TotalBsmtSF'] + X_test['1stFlrSF'] + X_test['2ndFlrSF']
    
    # Example feature: House Age
    if 'YearBuilt' in X_train.columns:
        current_year = 2023  # Update as needed
        X_train['HouseAge'] = current_year - X_train['YearBuilt']
        X_test['HouseAge'] = current_year - X_test['YearBuilt']
    
    # Example feature: Total Bathrooms
    bathroom_cols = [col for col in X_train.columns if 'Bath' in col]
    if bathroom_cols:
        X_train['TotalBathrooms'] = X_train[bathroom_cols].sum(axis=1)
        X_test['TotalBathrooms'] = X_test[bathroom_cols].sum(axis=1)
    
    return X_train, X_test

# Model Building and Evaluation
def build_models(X_train, X_test, y_train, y_test, preprocessor):
    # Create pipelines for different models
    models = {
        'Ridge': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(random_state=42))
        ]),
        'Lasso': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(random_state=42))
        ]),
        'ElasticNet': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet(random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(random_state=42))
        ]),
        'LightGBM': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(random_state=42))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # If we used log transform, convert back to original scale
        if 'SalePrice_Log' in y_test.name:
            y_pred_orig = np.expm1(y_pred)
            y_test_orig = np.expm1(y_test)
        else:
            y_pred_orig = y_pred
            y_test_orig = y_test
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        # Save results
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"{name} - RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, R²: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
        plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title(f'Actual vs Predicted Prices - {name}')
        plt.tight_layout()
        plt.savefig(f'prediction_plot_{name}.png')
    
    # Visualize model comparison
    model_names = list(results.keys())
    rmse_values = [results[model]['RMSE'] for model in model_names]
    r2_values = [results[model]['R2'] for model in model_names]
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE comparison
    sns.barplot(x=model_names, y=rmse_values, ax=ax[0])
    ax[0].set_title('RMSE by Model')
    ax[0].set_ylabel('RMSE ($)')
    ax[0].tick_params(axis='x', rotation=45)
    
    # R² comparison
    sns.barplot(x=model_names, y=r2_values, ax=ax[1])
    ax[1].set_title('R² by Model')
    ax[1].set_ylabel('R²')
    ax[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = models[best_model_name]
    
    return results, best_model_name, best_model

# Hyperparameter Tuning
def tune_best_model(X_train, y_train, best_model_name, preprocessor):
    # Define parameter grids for different models
    param_grids = {
        'Ridge': {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Lasso': {
            'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
        },
        'ElasticNet': {
            'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        'RandomForest': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7],
            'regressor__subsample': [0.7, 0.8, 0.9]
        },
        'LightGBM': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7],
            'regressor__num_leaves': [31, 50, 70]
        }
    }
    
    # Create the base model
    if best_model_name == 'Ridge':
        base_model = Ridge(random_state=42)
    elif best_model_name == 'Lasso':
        base_model = Lasso(random_state=42)
    elif best_model_name == 'ElasticNet':
        base_model = ElasticNet(random_state=42)
    elif best_model_name == 'RandomForest':
        base_model = RandomForestRegressor(random_state=42)
    elif best_model_name == 'GradientBoosting':
        base_model = GradientBoostingRegressor(random_state=42)
    elif best_model_name == 'XGBoost':
        base_model = xgb.XGBRegressor(random_state=42)
    else:  # LightGBM
        base_model = lgb.LGBMRegressor(random_state=42)
    
    # Create pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', base_model)
    ])
    
    # Perform grid search
    print(f"Tuning {best_model_name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[best_model_name],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.2f}")
    
    return grid_search.best_estimator_

# Feature Importance Analysis
def analyze_feature_importance(best_model, X_train, preprocessor):
    # Get feature names
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(columns))
        else:
            feature_names.extend(columns)
    
    # Models that support feature_importances_ attribute
    if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
        importances = best_model.named_steps['regressor'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
        plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Return top features
        top_features = [feature_names[i] for i in indices[:10]]
        return top_features
    
    # Linear models
    elif hasattr(best_model.named_steps['regressor'], 'coef_'):
        coefs = best_model.named_steps['regressor'].coef_
        
        # Get absolute coefficients and sort
        abs_coefs = np.abs(coefs)
        indices = np.argsort(abs_coefs)[::-1]
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        plt.title('Feature Coefficients (Absolute Values)')
        plt.bar(range(min(20, len(indices))), abs_coefs[indices[:20]], align='center')
        plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_coefficients.png')
        
        # Return top features
        top_features = [feature_names[i] for i in indices[:10]]
        return top_features
    
    return None

# Data Visualization for PowerBI/Tableau
def prepare_visualization_data(df, best_model, top_features, preprocessor, X_test, y_test):
    """Prepare data for visualization dashboards"""
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # If log transformed, convert back
    if 'SalePrice_Log' in df.columns:
        y_pred_orig = np.expm1(y_pred)
        y_test_orig = np.expm1(y_test)
    else:
        y_pred_orig = y_pred
        y_test_orig = y_test
    
    # Create prediction results dataframe
    pred_df = pd.DataFrame({
        'Actual_Price': y_test_orig,
        'Predicted_Price': y_pred_orig,
        'Absolute_Error': np.abs(y_test_orig - y_pred_orig),
        'Percentage_Error': np.abs((y_test_orig - y_pred_orig) / y_test_orig) * 100
    })
    
    # Add key features to prediction results
    if top_features:
        test_features = X_test.reset_index(drop=True)
        common_features = [f for f in top_features if f in test_features.columns]
        for feature in common_features:
            pred_df[feature] = test_features[feature].values
    
    # Save for visualization tools
    pred_df.to_csv('prediction_results.csv', index=False)
    
    # Create aggregated data for neighborhood analysis
    if 'Neighborhood' in X_test.columns:
        neighborhood_data = X_test.join(pd.Series(y_test_orig, index=X_test.index, name='Actual_Price'))
        neighborhood_data = neighborhood_data.join(pd.Series(y_pred_orig, index=X_test.index, name='Predicted_Price'))
        
        neighborhood_summary = neighborhood_data.groupby('Neighborhood').agg({
            'Actual_Price': ['mean', 'median', 'count'],
            'Predicted_Price': ['mean']
        })
        
        neighborhood_summary.columns = ['Avg_Actual_Price', 'Median_Actual_Price', 'Count', 'Avg_Predicted_Price']
        neighborhood_summary['Price_Per_Sqft'] = neighborhood_summary['Avg_Actual_Price'] / neighborhood_data.groupby('Neighborhood')['GrLivArea'].mean()
        
        neighborhood_summary.to_csv('neighborhood_analysis.csv')
    
    # Create error analysis for model interpretability
    error_df = pred_df.copy()
    error_df['Error'] = y_test_orig - y_pred_orig
    error_df['Error_Category'] = pd.cut(
        error_df['Error'], 
        bins=5,
        labels=['Highly Underestimated', 'Underestimated', 'Accurate', 'Overestimated', 'Highly Overestimated']
    )
    
    error_df.to_csv('error_analysis.csv', index=False)
    
    return pred_df

# Main function
def main():
    # Load and explore data
    df = load_data()
    df = explore_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Feature engineering
    X_train, X_test = feature_engineering(X_train, X_test)
    
    # Build and evaluate models
    results, best_model_name, best_model = build_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Tune best model
    tuned_model = tune_best_model(X_train, y_train, best_model_name, preprocessor)
    
    # Analyze feature importance
    top_features = analyze_feature_importance(tuned_model, X_train, preprocessor)
    
    # Prepare data for visualization
    pred_df = prepare_visualization_data(df, tuned_model, top_features, preprocessor, X_test, y_test)
    
    print("\nAnalysis complete. Files generated for PowerBI/Tableau visualizations.")
    print("- prediction_results.csv: Contains actual vs predicted prices")
    print("- neighborhood_analysis.csv: Aggregated data by neighborhood")
    print("- error_analysis.csv: Error categorization for model interpretability")
    print("\nVisualization plots saved:")
    print("- price_distribution.png: Distribution of house prices")
    print("- top_correlations.png: Features most correlated with price")
    print("- feature_importance.png: Most important features for prediction")
    print("- model_comparison.png: Performance comparison of different models")

if __name__ == "__main__":
    main()