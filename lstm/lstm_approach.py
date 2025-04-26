"""
LSTM Approach for Bitcoin Price Prediction (Fixed Version)

This script implements an LSTM-based model for predicting Bitcoin price movements
using sequential price and technical indicator data, with fixes for data type issues.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys

# Add the main directory to path to import utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def prepare_lstm_data(data, window_size=60, prediction_horizon=1, test_size=0.2):
    """
    Prepare data for LSTM model training
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLCV and technical indicators
    window_size (int): Number of time steps to use for each prediction
    prediction_horizon (int): Number of steps ahead to predict
    test_size (float): Proportion of data to use for testing
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, scalers)
    """
    # Select relevant features
    features = ['Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
    target = 'Target'
    
    # Check if all features are in the dataframe
    available_features = [f for f in features if f in data.columns]
    if len(available_features) < len(features):
        print(f"Warning: Some features are missing. Using available features: {available_features}")
        features = available_features
    
    if target not in data.columns:
        print("Target column not found. Creating target...")
        data['Target'] = (data['Close'].shift(-prediction_horizon) > data['Close']).astype(int)
    
    # Scale the features
    feature_scalers = {}
    scaled_features = pd.DataFrame(index=data.index)
    
    for feature in features:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
        feature_scalers[feature] = scaler
    
    # Prepare sequences
    X, y = [], []
    for i in range(window_size, len(scaled_features) - prediction_horizon):
        X.append(scaled_features.iloc[i-window_size:i].values)
        y.append(data[target].iloc[i])
    
    X, y = np.array(X), np.array(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, feature_scalers

def create_lstm_model(input_shape):
    """
    Create and compile LSTM model for price prediction
    
    Parameters:
    input_shape (tuple): Input shape (time steps, features)
    
    Returns:
    keras.Model: Compiled LSTM model
    """
    model = Sequential([
        # First LSTM layer with return sequences for stacking
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(units=25, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def lstm_approach(data, window_size=60, prediction_horizon=1, epochs=50, batch_size=32, validation_split=0.1):
    """
    Train and evaluate an LSTM model for Bitcoin price prediction
    
    Parameters:
    data (pandas.DataFrame): OHLCV data with technical indicators
    window_size (int): Number of time steps for each prediction
    prediction_horizon (int): Number of steps ahead to predict
    epochs (int): Number of training epochs
    batch_size (int): Batch size for training
    validation_split (float): Fraction of training data to use for validation
    
    Returns:
    tuple: (model, history, evaluation_metrics)
    """
    print(f"Running LSTM approach with window size {window_size}...")
    
    # Convert data to numeric
    numeric_data = data.copy()
    for col in numeric_data.columns:
        if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    numeric_data = numeric_data.dropna()
    
    # Add technical indicators
    data_with_indicators = add_technical_indicators(numeric_data)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scalers = prepare_lstm_data(
        data_with_indicators, 
        window_size=window_size, 
        prediction_horizon=prediction_horizon,
        test_size=0.2
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    print(model.summary())
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Calculate evaluation metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Save model
    model.save('models/lstm_model.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/lstm_history.png')
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    
    # Get the prediction probabilities
    y_pred_prob = model.predict(X_test).flatten()
    
    # Plot the last 100 predictions
    last_n = 100
    plt.subplot(2, 1, 1)
    plt.plot(y_test[-last_n:], label='Actual', marker='o', markersize=3)
    plt.plot(y_pred[-last_n:], label='Predicted', marker='x', markersize=3)
    plt.title('LSTM Model - Actual vs Predicted (Last 100 Days)')
    plt.ylabel('Direction (1=Up, 0=Down)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot distribution of prediction probabilities
    plt.subplot(2, 1, 2)
    plt.hist(y_pred_prob, bins=20, alpha=0.7)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/lstm_predictions.png')
    
    return model, history, metrics

if __name__ == "__main__":
    # Load data
    try:
        data = pd.read_csv('data/bitcoin_historical.csv', index_col=0)
        print(f"Loaded data from CSV. Shape: {data.shape}")
    except FileNotFoundError:
        # Import only if needed
        from btc_prediction_toolkit import fetch_bitcoin_data
        data = fetch_bitcoin_data()
    
    # Run LSTM approach
    model, history, metrics = lstm_approach(
        data,
        window_size=60,
        prediction_horizon=1,
        epochs=50,
        batch_size=32
    )
    
    print("\nLSTM Approach Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])