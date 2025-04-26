"""
ARIMA Approach for Bitcoin Price Prediction

This script implements ARIMA (AutoRegressive Integrated Moving Average)
for time series forecasting of Bitcoin prices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import pickle
import warnings
import sys

# Add the main directory to path to import utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btc_prediction_toolkit import fetch_bitcoin_data, evaluate_model

# Suppress warnings
warnings.filterwarnings("ignore")

def check_stationarity(series, window=12):
    """
    Check if a time series is stationary using ADF test and rolling statistics
    
    Parameters:
    series (pandas.Series): Time series to check
    window (int): Window size for rolling statistics
    
    Returns:
    dict: Results of stationarity test
    """
    # Rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Augmented Dickey-Fuller test
    adf_result = adfuller(series.dropna())
    
    # Create a results dictionary
    results = {
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'is_stationary': adf_result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
    }
    
    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.title('Rolling Mean & Standard Deviation')
    plt.plot(series, label='Original')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Standard Deviation')
    plt.legend()
    plt.savefig('results/arima_stationarity_check.png')
    
    return results

def find_best_arima_parameters(series):
    """
    Automatically find the best ARIMA parameters using auto_arima
    
    Parameters:
    series (pandas.Series): Time series data
    
    Returns:
    tuple: Best order (p, d, q) and seasonal order (P, D, Q, s)
    """
    print("Finding best ARIMA parameters (this may take a while)...")
    
    # Use auto_arima to find the best parameters
    model = auto_arima(
        series,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,  # Let auto_arima determine 'd'
        seasonal=False,  # No seasonality for daily Bitcoin data
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    # Get the order (p, d, q)
    best_order = model.order
    
    return best_order

def arima_forecast(series, test_size=0.2, forecast_periods=30, retrain_frequency=7):
    """
    Train ARIMA model and make forecasts
    
    Parameters:
    series (pandas.Series): Time series data
    test_size (float): Proportion of data to use for testing
    forecast_periods (int): Number of periods to forecast at each step
    retrain_frequency (int): How often to retrain the model (in periods)
    
    Returns:
    tuple: (predictions, actuals, model)
    """
    # Split data into train and test
    split_idx = int(len(series) * (1 - test_size))
    train = series[:split_idx]
    test = series[split_idx:]
    
    print(f"Training data size: {len(train)}")
    print(f"Testing data size: {len(test)}")
    
    # Check stationarity of training data
    stationarity_results = check_stationarity(train)
    print(f"Training data stationarity check:")
    print(f"  ADF Statistic: {stationarity_results['adf_statistic']:.4f}")
    print(f"  p-value: {stationarity_results['p_value']:.4f}")
    print(f"  Is stationary: {stationarity_results['is_stationary']}")
    
    # Find best ARIMA parameters if not stationary, or use default
    if not stationarity_results['is_stationary']:
        print("Data is not stationary, finding optimal parameters...")
        order = find_best_arima_parameters(train)
    else:
        order = (1, 0, 1)  # Default ARIMA(1,0,1) for stationary data
    
    print(f"Using ARIMA{order} model")
    
    # Initialize predictions array
    predictions = []
    
    # Walk-forward validation
    history = [x for x in train]
    
    for i in range(0, len(test), retrain_frequency):
        # Get the slice of test data to predict
        end_idx = min(i + retrain_frequency, len(test))
        test_slice = test[i:end_idx]
        
        # Train model on history
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        
        # Make forecast
        forecast = model_fit.forecast(steps=len(test_slice))
        
        # Convert forecast to binary predictions (up or down)
        binary_forecast = []
        for j in range(len(forecast)):
            if j == 0:
                # First prediction is compared to last point in history
                binary_forecast.append(1 if forecast[j] > history[-1] else 0)
            else:
                # Subsequent predictions are compared to previous forecast
                binary_forecast.append(1 if forecast[j] > forecast[j-1] else 0)
        
        # Add predictions to list
        predictions.extend(binary_forecast)
        
        # Update history
        history.extend(test_slice)
    
    # Ensure predictions and actuals are the same length
    predictions = predictions[:len(test)]
    
    # Convert actuals to binary (1 if up, 0 if down)
    actuals = []
    for i in range(1, len(test) + 1):
        if i == 1:
            # First actual is compared to last point in train
            actuals.append(1 if test.iloc[i-1] > train.iloc[-1] else 0)
        else:
            # Subsequent actuals are compared to previous actual
            actuals.append(1 if test.iloc[i-1] > test.iloc[i-2] else 0)
    
    return predictions, actuals, model_fit

def arima_approach(data, target_col='Close', test_size=0.2):
    """
    Train and evaluate an ARIMA model for Bitcoin price prediction
    
    Parameters:
    data (pandas.DataFrame): OHLCV data
    target_col (str): Column to predict
    test_size (float): Proportion of data to use for testing
    
    Returns:
    tuple: (model, predictions, actuals, metrics)
    """
    print(f"Running ARIMA approach for predicting {target_col}...")
    
    # Get time series
    series = data[target_col]
    
    # Train and forecast
    predictions, actuals, model = arima_forecast(
        series, 
        test_size=test_size,
        forecast_periods=30,
        retrain_frequency=7
    )
    
    # Evaluate model
    metrics = evaluate_model(actuals, predictions)
    
    # Plot predictions vs actuals
    plt.figure(figsize=(15, 6))
    
    # Plot the last 100 predictions
    last_n = 100
    plt.plot(actuals[-last_n:], label='Actual', marker='o', markersize=3)
    plt.plot(predictions[-last_n:], label='Predicted', marker='x', markersize=3)
    plt.title(f'ARIMA Model - Actual vs Predicted Direction (Last {last_n} Days)')
    plt.ylabel('Direction (1=Up, 0=Down)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/arima_predictions.png')
    
    # Save model
    with open('models/arima_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, predictions, actuals, metrics

if __name__ == "__main__":
    # Load data
    try:
        data = pd.read_csv('data/bitcoin_historical.csv', index_col=0)
        print(f"Loaded data from CSV. Shape: {data.shape}")
    except FileNotFoundError:
        data = fetch_bitcoin_data()
    
    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)
    
    # Run ARIMA approach
    model, predictions, actuals, metrics = arima_approach(
        data,
        target_col='Close',
        test_size=0.2
    )
    
    print("\nARIMA Approach Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])