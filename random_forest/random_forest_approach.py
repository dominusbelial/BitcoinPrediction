"""
Random Forest Approach for Bitcoin Price Prediction

This script implements Random Forest, an ensemble learning algorithm,
for Bitcoin price prediction using technical indicators.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import sys

# Add the main directory to path to import utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btc_prediction_toolkit import fetch_bitcoin_data, add_technical_indicators, evaluate_model

def prepare_features(data, window_size=10, prediction_horizon=1):
    """
    Prepare features for Random Forest model including lagged features
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLCV and technical indicators
    window_size (int): Number of lagged features to include
    prediction_horizon (int): Number of periods ahead to predict
    
    Returns:
    tuple: (X, y, feature_names)
    """
    # Drop NaN values
    data = data.copy().dropna()
    
    # Create target variable (1 if price goes up, 0 if down)
    data['Target'] = (data['Close'].shift(-prediction_horizon) > data['Close']).astype(int)
    
    # Select base features (excluding the target and date)
    base_features = [col for col in data.columns if col not in ['Target', 'Volume', 'Adj Close']]
    
    # Remove non-numerical columns
    for col in base_features.copy():
        if not np.issubdtype(data[col].dtype, np.number):
            base_features.remove(col)
    
    # Create lagged features
    for feature in base_features:
        for lag in range(1, window_size + 1):
            data[f"{feature}_lag_{lag}"] = data[feature].shift(lag)
    
    # Drop NaN values after creating lagged features
    data = data.dropna()
    
    # Get all features including lagged ones
    all_features = [col for col in data.columns if col != 'Target' and np.issubdtype(data[col].dtype, np.number)]
    
    # Features and target
    X = data[all_features].iloc[:-prediction_horizon]  # Remove last rows without target
    y = data['Target'].iloc[:-prediction_horizon]
    
    return X, y, all_features

def train_random_forest_model(X_train, y_train, param_grid=None, cv=5):
    """
    Train Random Forest model with optional hyperparameter tuning
    
    Parameters:
    X_train (pandas.DataFrame): Training features
    y_train (pandas.Series): Training targets
    param_grid (dict): Hyperparameters to search
    cv (int): Number of cross-validation folds
    
    Returns:
    sklearn.ensemble.RandomForestClassifier: Trained model
    """
    if param_grid is None:
        # Default parameters
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train, y_train)
    else:
        # Grid search for hyperparameter tuning
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        
        model = grid_search.best_estimator_
    
    return model

def feature_importance_analysis(model, feature_names):
    """
    Analyze and visualize feature importances for Random Forest
    
    Parameters:
    model (sklearn.ensemble.RandomForestClassifier): Trained Random Forest model
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
    
    # Limit to the top 30 features for readability
    top_n = min(30, len(importances))
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
    plt.tight_layout()
    plt.savefig('results/random_forest_feature_importance.png')
    
    # Create feature importance dictionary
    importance_dict = {feature_names[i]: importances[i] for i in range(len(feature_names))}
    
    return importance_dict

def random_forest_approach(data, window_size=10, prediction_horizon=1, test_size=0.2, tune_hyperparams=False):
    """
    Train and evaluate a Random Forest model for Bitcoin price prediction
    
    Parameters:
    data (pandas.DataFrame): OHLCV data with technical indicators
    window_size (int): Number of lagged features to include
    prediction_horizon (int): Number of periods ahead to predict
    test_size (float): Proportion of data to use for testing
    tune_hyperparams (bool): Whether to perform hyperparameter tuning
    
    Returns:
    tuple: (model, metrics, feature_importances)
    """
    print(f"Running Random Forest approach with window size {window_size}...")
    
    # Prepare features
    X, y, feature_names = prepare_features(data, window_size, prediction_horizon)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    if tune_hyperparams:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = train_random_forest_model(X_train_scaled, y_train, param_grid)
    else:
        model = train_random_forest_model(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred)
    
    # Feature importance analysis
    feature_importances = feature_importance_analysis(model, feature_names)
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    
    # Plot the last 100 predictions
    last_n = 100
    
    # Get only the last n values
    y_test_last = y_test[-last_n:].values if len(y_test) > last_n else y_test.values
    y_pred_last = y_pred[-last_n:] if len(y_pred) > last_n else y_pred
    
    # Get prediction probabilities
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_prob_last = y_pred_prob[-last_n:] if len(y_pred_prob) > last_n else y_pred_prob
    
    # Plot predictions
    plt.subplot(2, 1, 1)
    plt.plot(y_test_last, label='Actual', marker='o', markersize=3)
    plt.plot(y_pred_last, label='Predicted', marker='x', markersize=3)
    plt.title(f'Random Forest Model - Actual vs Predicted (Last {len(y_test_last)} Days)')
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
    plt.savefig('results/random_forest_predictions.png')
    
    # Save model and scaler
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/random_forest_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
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
    
    # Run Random Forest approach
    model, metrics, feature_importances = random_forest_approach(
        data,
        window_size=10,
        prediction_horizon=1,
        test_size=0.2,
        tune_hyperparams=False  # Set to True to perform hyperparameter tuning
    )
    
    print("\nRandom Forest Approach Evaluation:")
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