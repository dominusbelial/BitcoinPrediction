"""
Visual CNN Approach for Bitcoin Price Prediction (Fixed Version)

This script implements a CNN-based approach that uses candlestick chart images
to predict Bitcoin price movements, with fixes for data type issues.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import io
from PIL import Image
import sys

# Add the main directory to path to import utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CandlestickImageGenerator(Sequence):
    """Generate batches of candlestick images"""
    
    def __init__(self, df, window_size=60, prediction_horizon=1, batch_size=32, 
                 img_width=128, img_height=128, shuffle=True):
        """
        Initialize the generator
        
        Parameters:
        df (pandas.DataFrame): OHLCV data
        window_size (int): Number of candles in each image
        prediction_horizon (int): How many periods ahead to predict
        batch_size (int): Batch size for training
        img_width, img_height (int): Dimensions of output images
        shuffle (bool): Whether to shuffle data on epoch end
        """
        # Make sure the data is numeric
        self.df = df.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.shuffle = shuffle
        self.n_samples = len(self.df) - window_size - prediction_horizon
        self.indexes = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        """Return the number of batches"""
        return int(np.floor(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        """Return a batch of data"""
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X = np.empty((self.batch_size, self.img_height, self.img_width, 1))
        y = np.empty(self.batch_size, dtype=int)
        
        for i, index in enumerate(batch_indexes):
            # Get window of data
            window_data = self.df.iloc[index:index + self.window_size]
            
            # Generate candlestick image
            img = self._generate_candlestick_image(window_data)
            X[i] = img
            
            # Get label (1 if price goes up, 0 if down)
            current_close = self.df.iloc[index + self.window_size - 1]['Close']
            future_close = self.df.iloc[index + self.window_size + self.prediction_horizon - 1]['Close']
            y[i] = 1 if future_close > current_close else 0
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indexes at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _generate_candlestick_image(self, data):
        """
        Generate a candlestick chart image from OHLCV data
        
        Parameters:
        data (pandas.DataFrame): OHLCV data for the window
        
        Returns:
        numpy.ndarray: Normalized grayscale image [height, width, 1]
        """
        # Create figure with no margins
        fig = Figure(figsize=(self.img_width/100, self.img_height/100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        
        # Hide axes
        ax.set_axis_off()
        
        # Convert to proper format for charting
        data = data.reset_index()  # Reset index so we can use integer positions
        
        # Create candlestick chart
        width = 0.8
        up = data[data.Close >= data.Open]
        down = data[data.Close < data.Open]
        
        # Plot up candles (only if there are any)
        if not up.empty:
            ax.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='white')
            ax.bar(up.index, up.High - up.Close, width/5, bottom=up.Close, color='white')
            ax.bar(up.index, up.Open - up.Low, width/5, bottom=up.Low, color='white')
        
        # Plot down candles (only if there are any)
        if not down.empty:
            ax.bar(down.index, down.Open - down.Close, width, bottom=down.Close, color='black')
            ax.bar(down.index, down.High - down.Open, width/5, bottom=down.Open, color='black')
            ax.bar(down.index, down.Close - down.Low, width/5, bottom=down.Low, color='black')
        
        # Set tight layout
        fig.tight_layout(pad=0)
        
        # Convert to image
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_rgba(buf)
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        # Convert to grayscale
        img = np.mean(img[:, :, :3], axis=2)
        img = img.reshape(img.shape + (1,))
        
        # Normalize
        img = img / 255.0
        
        # Resize if necessary
        if img.shape[0] != self.img_height or img.shape[1] != self.img_width:
            img = np.array(Image.fromarray((img * 255).astype(np.uint8).reshape(img.shape[:2]))
                          .resize((self.img_width, self.img_height))) / 255.0
            img = img.reshape(img.shape + (1,))
        
        return img

def create_cnn_model(input_shape):
    """
    Create and compile CNN model for image classification
    
    Parameters:
    input_shape (tuple): Input shape (height, width, channels)
    
    Returns:
    keras.Model: Compiled CNN model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def visual_cnn_approach(data, window_size=60, prediction_horizon=1, img_size=(128, 128),
                       batch_size=32, epochs=20, validation_split=0.2):
    """
    Train and evaluate a visual CNN model for Bitcoin price prediction
    
    Parameters:
    data (pandas.DataFrame): OHLCV data
    window_size (int): Number of candles in each image
    prediction_horizon (int): How many periods ahead to predict
    img_size (tuple): Image dimensions (width, height)
    batch_size (int): Batch size for training
    epochs (int): Number of training epochs
    validation_split (float): Fraction of data to use for validation
    
    Returns:
    tuple: (model, history, evaluation_metrics)
    """
    print(f"Running Visual CNN approach with window size {window_size}...")
    
    # Make sure data has proper types
    numeric_data = data.copy()
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in numeric_data.columns:
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    numeric_data = numeric_data.dropna()
    
    # Split data into training and testing
    split_idx = int(len(numeric_data) * (1 - validation_split))
    train_data = numeric_data.iloc[:split_idx]
    val_data = numeric_data.iloc[split_idx - window_size - prediction_horizon:]
    
    print(f"Training samples: {len(train_data) - window_size - prediction_horizon}")
    print(f"Validation samples: {len(val_data) - window_size - prediction_horizon}")
    
    # Create data generators
    train_gen = CandlestickImageGenerator(
        train_data, 
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size,
        img_width=img_size[0],
        img_height=img_size[1]
    )
    
    val_gen = CandlestickImageGenerator(
        val_data,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size,
        img_width=img_size[0],
        img_height=img_size[1]
    )
    
    # Create model
    model = create_cnn_model((img_size[1], img_size[0], 1))
    print(model.summary())
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate model on validation data
    val_pred = []
    val_true = []
    
    for i in range(len(val_gen)):
        X, y = val_gen[i]
        preds = model.predict(X)
        val_pred.extend((preds > 0.5).astype(int).flatten())
        val_true.extend(y)
    
    # Calculate evaluation metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(val_true, val_pred),
        'precision': precision_score(val_true, val_pred, zero_division=0),
        'recall': recall_score(val_true, val_pred, zero_division=0),
        'f1': f1_score(val_true, val_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(val_true, val_pred)
    }
    
    # Save model
    model.save('models/visual_cnn_model.h5')
    
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
    plt.savefig('results/visual_cnn_history.png')
    
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
    
    # Run CNN approach
    model, history, metrics = visual_cnn_approach(
        data,
        window_size=60,
        prediction_horizon=1,
        img_size=(128, 128),
        batch_size=32,
        epochs=20
    )
    
    print("\nVisual CNN Approach Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])