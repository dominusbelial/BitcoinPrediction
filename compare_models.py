"""
Bitcoin Price Prediction - Model Comparison

This script runs all implemented models and compares their performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import json
from datetime import datetime

# Add all directories to path
for dir_name in ['visual_cnn', 'lstm', 'arima', 'xgboost', 'random_forest']:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name))

# Import utility functions
from btc_prediction_toolkit import fetch_bitcoin_data, add_technical_indicators, plot_results

def load_module(script_path, module_name):
    """
    Dynamically load Python modules
    
    Parameters:
    script_path (str): Path to Python script
    module_name (str): Name to assign to the module
    
    Returns:
    module: Loaded Python module
    """
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_all_models(data):
    """
    Run all prediction models and compare results
    
    Parameters:
    data (pandas.DataFrame): OHLCV data
    
    Returns:
    dict: Results for all models
    """
    results = {}
    
    # Prepare data with technical indicators
    data_with_indicators = add_technical_indicators(data)
    
    # 1. Visual CNN Approach
    try:
        print("\n" + "="*50)
        print("Running Visual CNN Approach...")
        visual_cnn = load_module('visual_cnn/visual_cnn_approach.py', 'visual_cnn')
        model, history, metrics = visual_cnn.visual_cnn_approach(
            data,
            window_size=60,
            prediction_horizon=1,
            img_size=(128, 128),
            batch_size=32,
            epochs=10  # Reduced for faster execution
        )
        results['Visual CNN'] = metrics
        print("Visual CNN completed successfully.")
    except Exception as e:
        print(f"Error running Visual CNN: {e}")
    
    # 2. LSTM Approach
    try:
        print("\n" + "="*50)
        print("Running LSTM Approach...")
        lstm = load_module('lstm/lstm_approach.py', 'lstm')
        model, history, metrics = lstm.lstm_approach(
            data_with_indicators,
            window_size=60,
            prediction_horizon=1,
            epochs=20  # Reduced for faster execution
        )
        results['LSTM'] = metrics
        print("LSTM completed successfully.")
    except Exception as e:
        print(f"Error running LSTM: {e}")
    
    # 3. ARIMA Approach
    try:
        print("\n" + "="*50)
        print("Running ARIMA Approach...")
        arima = load_module('arima/arima_approach.py', 'arima')
        model, predictions, actuals, metrics = arima.arima_approach(
            data,
            target_col='Close',
            test_size=0.2
        )
        results['ARIMA'] = metrics
        print("ARIMA completed successfully.")
    except Exception as e:
        print(f"Error running ARIMA: {e}")
    
    # 4. XGBoost Approach
    try:
        print("\n" + "="*50)
        print("Running XGBoost Approach...")
        xgboost = load_module('xgboost/xgboost_approach.py', 'xgboost')
        model, metrics, feature_importances = xgboost.xgboost_approach(
            data_with_indicators,
            prediction_horizon=1,
            test_size=0.2,
            tune_hyperparams=False
        )
        results['XGBoost'] = metrics
        print("XGBoost completed successfully.")
    except Exception as e:
        print(f"Error running XGBoost: {e}")
    
    # 5. Random Forest Approach
    try:
        print("\n" + "="*50)
        print("Running Random Forest Approach...")
        random_forest = load_module('random_forest/random_forest_approach.py', 'random_forest')
        model, metrics, feature_importances = random_forest.random_forest_approach(
            data_with_indicators,
            window_size=10,
            prediction_horizon=1,
            test_size=0.2,
            tune_hyperparams=False
        )
        results['Random Forest'] = metrics
        print("Random Forest completed successfully.")
    except Exception as e:
        print(f"Error running Random Forest: {e}")
    
    return results

def save_results(results):
    """
    Save model comparison results to file
    
    Parameters:
    results (dict): Results for all models
    """
    # Create a results summary
    summary = {}
    for model_name, metrics in results.items():
        summary[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
    
    # Save to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/model_comparison_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Create a summary table
    results_df = pd.DataFrame.from_dict({
        model: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1']
        }
        for model, metrics in results.items()
    }, orient='index')
    
    # Save to CSV
    results_df.to_csv(f'results/model_comparison_{timestamp}.csv')
    
    # Print table
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(results_df)
    print("\nResults saved to:", f'results/model_comparison_{timestamp}.csv')
    
    return results_df

if __name__ == "__main__":
    # Create directories if they don't exist
    for dir_name in ['models', 'results', 'data', 'visual_cnn', 'lstm', 'arima', 'xgboost', 'random_forest']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Load or fetch data
    try:
        data = pd.read_csv('data/bitcoin_historical.csv', index_col=0, parse_dates=True)
        print(f"Loaded data from CSV. Shape: {data.shape}")
    except FileNotFoundError:
        print("Fetching Bitcoin historical data...")
        data = fetch_bitcoin_data()
    
    # Run all models
    results = run_all_models(data)
    
    # Save and display results
    results_df = save_results(results)
    
    # Plot comparison
    plot_results(results, title='Bitcoin Price Prediction Model Comparison')