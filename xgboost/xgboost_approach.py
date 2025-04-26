"""
XGBoost Approach for Bitcoin Price Prediction

This script implements XGBoost, a gradient boosting algorithm,
for Bitcoin price prediction using technical indicators.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import shap

# Add the main directory to path to import utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btc_prediction_toolkit import fetch_bitcoin_data, add_technical_indicators, evaluate_model

def prepare_features(data, prediction_horizon=1):
    """
    Prepare features for XGBoost model
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLCV and technical indicators
    prediction_horizon (int): Number of periods ahead to predict
    
    Returns:
    tuple: (X, y, feature_names)
    """
    # Drop NaN values
    data = data.copy().dropna()
    
    # Create target variable (1 if price goes up, 0 if down)
    data['Target'] = (data['Close'].shift(-prediction_horizon) > data['Close']).astype(int)
    
    # Select features (excluding the target and date)
    features = [col for col in data.columns if col not in ['Target', 'Volume', 'Adj Close']]
    
    # Remove non-numerical columns
    for col in features.copy():
        if not np.issubdtype(data[col].dtype, np.number):
            features.remove(col)
    
    # Features and target
    X = data[features].iloc[:-prediction_horizon]  # Remove last rows without target
    y = data['Target'].iloc[:-prediction_horizon]
    
    return X, y, features

def train_xgboost_model(X_train, y_train, param_grid=None, cv=5):
    """
    Train XGBoost model with optional hyperparameter tuning
    
    Parameters:
    X_train (pandas.DataFrame): Training features
    y_train (pandas.Series): Training targets
    param_grid (dict): Hyperparameters to search
    cv (int): Number of cross-validation folds
    
    Returns:
    xgboost.XGBClassifier: Trained model
    """
    if param_grid is None:
        # Default parameters
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42
        )
        model.fit(X_train, y_train)
    else:
        # Grid search for hyperparameter tuning
        base_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        
        model = grid_search.best_estimator_
    
    return model

def feature_importance_analysis(model, X, feature_names):
    """
    Analyze and visualize feature importances
    
    Parameters:
    model (xgboost.XGBClassifier): Trained XGBoost model
    X (pandas.DataFrame): Feature data
    feature_names (list): Names of features
    
    Returns:
    dict: Feature importance scores
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('results/xgboost_feature_importance.png')
    
    # Create feature importance dictionary
    importance_dict = {feature_names[i]: importances[i] for i in range(len(feature_names))}
    
    # SHAP values for feature importance
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('results/xgboost_shap_importance.png')
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
    
    return importance_dict

def xgboost_approach(data, prediction_horizon=1, test_size=0.2, tune_hyperparams=False):
    """
    Train and evaluate an XGBoost model for Bitcoin price prediction
    
    Parameters:
    data (pandas.DataFrame): OHLCV data with technical indicators
    prediction_horizon (int): Number of periods ahead to predict
    test_size (float): Proportion of data to use for testing
    tune_hyperparams (bool): Whether to perform hyperparameter tuning
    
    Returns:
    tuple: (model, metrics, feature_importances)
    """
    print(f"Running XGBoost approach for {prediction_horizon}-day ahead prediction...")
    
    # Prepare features
    X, y, feature_names = prepare_features(data, prediction_horizon)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    if tune_hyperparams:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        model = train_xgboost_model(X_train, y_train, param_grid)
    else:
        model = train_xgboost_model(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred)
    
    # Analyze feature importance
    feature_importances = feature_importance_analysis(model, X, feature_names)
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    
    # Plot the last 100 predictions
    last_n = 100
    
    # Get only the last n values
    y_test_last = y_test[-last_n:].values if len(y_test) > last_n else y_test.values
    y_pred_last = y_pred[-last_n:] if len(y_pred) > last_n else y_pred
    
    # Get prediction probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_prob_last = y_pred_prob[-last_n:] if len(y_pred_prob) > last_n else y_pred_prob
    
    # Plot predictions
    plt.subplot(2, 1, 1)
    plt.plot(y_test_last, label='Actual', marker='o', markersize=3)
    plt.plot(y_pred_last, label='Predicted', marker='x', markersize=3)
    plt.title(f'XGBoost Model - Actual vs Predicted (Last {len(y_test_last)} Days)')
    plt.ylabel('Direction (1=Up, 0=Down)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot prediction probabilities
    plt.subplot(2, 1, 2)
    plt.bar(range(len(y_pred_prob_last)), y_pred_prob_last, alpha=0.6)
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.title('Prediction Probabilities')
    plt.xlabel('Day')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/xgboost_predictions.png')
    
    # Save model
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, metrics, feature_importances

if __name__ == "__main__":
    # Load data
    try:
        data = pd.read_csv('data/bitcoin_historical.csv', index_col=0)
        print(f"Loaded data from CSV. Shape: {data.shape}")
    except FileNotFoundError:
        data = fetch_bitcoin_data()
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Run XGBoost approach
    model, metrics, feature_importances = xgboost_approach(
        data,
        prediction_horizon=1,
        test_size=0.2,
        tune_hyperparams=False  # Set to True to perform hyperparameter tuning
    )
    
    print("\nXGBoost Approach Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("\nTop 5 Important Features:")
    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")