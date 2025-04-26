"""
Bitcoin Price Prediction Toolkit (Fixed Version)

This package implements various approaches for predicting Bitcoin price movements:
1. Visual CNN - Using candlestick charts as images
2. LSTM - Time series prediction using Long Short-Term Memory networks
3. ARIMA - Statistical time series forecasting
4. XGBoost - Gradient boosting with technical indicators
5. Random Forest - Ensemble learning with technical indicators
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create directories for models and data
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def fetch_bitcoin_data(start_date='2015-01-01', end_date=None):
    """
    Fetch Bitcoin historical data from Yahoo Finance
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
    
    Returns:
    pandas.DataFrame: Bitcoin OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    print(f"Fetching Bitcoin data from {start_date} to {end_date}...")
    ticker = "BTC-USD"
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Save to CSV
    data.to_csv('data/bitcoin_historical.csv')
    print(f"Data saved to 'data/bitcoin_historical.csv'. Shape: {data.shape}")
    
    return data

def add_technical_indicators(df):
    """
    Add common technical indicators to the dataframe
    
    Parameters:
    df (pandas.DataFrame): OHLCV dataframe
    
    Returns:
    pandas.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert columns to numeric
    for col in df.columns:
        if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    df = df.dropna()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # Price Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    
    # Labels for prediction: 1 if tomorrow's close is higher than today's, 0 otherwise
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance with various metrics
    
    Parameters:
    y_true (array): True labels
    y_pred (array): Predicted labels
    
    Returns:
    dict: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics

def plot_results(results, title='Model Performance Comparison'):
    """
    Plot comparative results of different models
    
    Parameters:
    results (dict): Dictionary with model names as keys and evaluation metrics as values
    title (str): Title for the plot
    """
    models = list(results.keys())
    accuracy = [results[model]['accuracy'] for model in models]
    precision = [results[model]['precision'] for model in models]
    recall = [results[model]['recall'] for model in models]
    f1 = [results[model]['f1'] for model in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width*1.5, accuracy, width, label='Accuracy')
    ax.bar(x - width/2, precision, width, label='Precision')
    ax.bar(x + width/2, recall, width, label='Recall')
    ax.bar(x + width*1.5, f1, width, label='F1 Score')
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.show()
    
    return

def inspect_dataframe(df, n_rows=5):
    """
    Inspect a dataframe and provide information about its structure and data types
    
    Parameters:
    df (pandas.DataFrame): DataFrame to inspect
    n_rows (int): Number of rows to display
    
    Returns:
    None
    """
    print(f"DataFrame shape: {df.shape}")
    print("\nDataFrame head:")
    print(df.head(n_rows))
    print("\nDataFrame info:")
    print(df.info())
    print("\nDataFrame data types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Check if there are any non-numeric values in typically numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            # Try to convert and see if we get any errors
            try:
                pd.to_numeric(df[col])
                print(f"Column {col} can be converted to numeric.")
            except ValueError:
                print(f"WARNING: Column {col} contains non-numeric values!")
                # Show some examples of non-numeric values
                non_numeric_values = df[pd.to_numeric(df[col], errors='coerce').isna()][col].unique()
                print(f"Examples of non-numeric values: {non_numeric_values[:5]}")
    
    return

if __name__ == "__main__":
    # Example usage
    data = fetch_bitcoin_data()
    inspect_dataframe(data)
    data_with_indicators = add_technical_indicators(data)
    print("Added technical indicators. New shape:", data_with_indicators.shape)