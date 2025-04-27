"""
Parallelized Visual CNN Approach for Bitcoin Price Prediction

This script implements a CNN-based approach that uses candlestick chart images
to predict Bitcoin price movements, with parallelization optimizations to improve training speed.
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math

# Enable mixed precision training for faster computation on compatible GPUs
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except:
    print("Mixed precision not available")

# Configure TensorFlow to use all available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Allow memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set visible devices if you have multiple GPUs and want to use specific ones
        # tf.config.experimental.set_visible_devices(gpus[:2], 'GPU')  # Use first two GPUs
        
        print(f"Using {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPUs found, using CPU")

# Optimize TensorFlow performance
tf.config.threading.set_inter_op_parallelism_threads(min(16, multiprocessing.cpu_count()))
tf.config.threading.set_intra_op_parallelism_threads(min(16, multiprocessing.cpu_count()))

# Add the main directory to path to import utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CandlestickImageGenerator(Sequence):
    """Generate batches of candlestick images with parallel processing"""
    
    def __init__(self, df, window_size=60, prediction_horizon=1, batch_size=32, 
                 img_width=128, img_height=128, shuffle=True, num_workers=None):
        """
        Initialize the generator
        
        Parameters:
        df (pandas.DataFrame): OHLCV data
        window_size (int): Number of candles in each image
        prediction_horizon (int): How many periods ahead to predict
        batch_size (int): Batch size for training
        img_width, img_height (int): Dimensions of output images
        shuffle (bool): Whether to shuffle data on epoch end
        num_workers (int): Number of parallel workers for image generation
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
        
        # Set number of workers for parallel processing
        self.num_workers = num_workers if num_workers is not None else max(1, min(8, multiprocessing.cpu_count() // 2))
        
        # Pre-generate all images in parallel to speed up training
        self.use_cache = True
        self.image_cache = {}
        
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
        
        # Parallel processing for batch generation
        if self.use_cache:
            # If we're using cache, load images from cache
            for i, index in enumerate(batch_indexes):
                if index in self.image_cache:
                    X[i] = self.image_cache[index]
                else:
                    window_data = self.df.iloc[index:index + self.window_size]
                    X[i] = self._generate_candlestick_image(window_data)
                    self.image_cache[index] = X[i]
                
                # Get label (1 if price goes up, 0 if down)
                current_close = self.df.iloc[index + self.window_size - 1]['Close']
                future_close = self.df.iloc[index + self.window_size + self.prediction_horizon - 1]['Close']
                y[i] = 1 if future_close > current_close else 0
        else:
            # Process images in parallel without caching
            futures = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for i, index in enumerate(batch_indexes):
                    window_data = self.df.iloc[index:index + self.window_size]
                    futures.append((i, executor.submit(self._generate_candlestick_image, window_data)))
                    
                    # Get label (1 if price goes up, 0 if down)
                    current_close = self.df.iloc[index + self.window_size - 1]['Close']
                    future_close = self.df.iloc[index + self.window_size + self.prediction_horizon - 1]['Close']
                    y[i] = 1 if future_close > current_close else 0
                
                # Collect results
                for i, future in futures:
                    X[i] = future.result()
        
        return X, y
    
    def prefetch_images(self, max_samples=None):
        """Pre-generate and cache images to speed up training"""
        if max_samples is None:
            max_samples = self.n_samples
        
        print(f"Prefetching {max_samples} images...")
        sample_indexes = self.indexes[:max_samples]
        
        # Create batches for parallel processing
        batch_size = 100
        batches = [sample_indexes[i:i+batch_size] for i in range(0, len(sample_indexes), batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            print(f"Prefetching batch {batch_idx+1}/{len(batches)}...")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                for index in batch:
                    window_data = self.df.iloc[index:index + self.window_size]
                    futures[index] = executor.submit(self._generate_candlestick_image, window_data)
                
                # Collect results
                for index, future in futures.items():
                    self.image_cache[index] = future.result()
    
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
        
        # Close figure to release memory
        plt.close(fig)
        
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
    
    # Compile model with optimized parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def visual_cnn_approach(data, window_size=60, prediction_horizon=1, img_size=(128, 128),
                       batch_size=32, epochs=20, validation_split=0.2, num_workers=None,
                       use_prefetch=True, prefetch_limit=1000):
    """
    Train and evaluate a visual CNN model for Bitcoin price prediction with parallel processing
    
    Parameters:
    data (pandas.DataFrame): OHLCV data
    window_size (int): Number of candles in each image
    prediction_horizon (int): How many periods ahead to predict
    img_size (tuple): Image dimensions (width, height)
    batch_size (int): Batch size for training
    epochs (int): Number of training epochs
    validation_split (float): Fraction of data to use for validation
    num_workers (int): Number of parallel workers
    use_prefetch (bool): Whether to prefetch and cache images
    prefetch_limit (int): Maximum number of images to prefetch
    
    Returns:
    tuple: (model, history, evaluation_metrics)
    """
    print(f"Running Parallelized Visual CNN approach with window size {window_size}...")
    
    # Set number of workers based on available CPU cores
    if num_workers is None:
        num_workers = max(1, min(8, multiprocessing.cpu_count() // 2))
    print(f"Using {num_workers} worker threads")
    
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
        img_height=img_size[1],
        num_workers=num_workers
    )
    
    val_gen = CandlestickImageGenerator(
        val_data,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size,
        img_width=img_size[0],
        img_height=img_size[1],
        num_workers=num_workers
    )
    
    # Prefetch images if enabled
    if use_prefetch:
        train_gen.prefetch_images(min(prefetch_limit, len(train_gen) * batch_size))
        val_gen.prefetch_images(min(prefetch_limit, len(val_gen) * batch_size))
    
    # Create model
    model = create_cnn_model((img_size[1], img_size[0], 1))
    print(model.summary())
    
    # Calculate steps per epoch to ensure we process all data
    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # Train model with parallel processing
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=num_workers,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Evaluate model on validation data
    val_pred = []
    val_true = []
    
    for i in range(len(val_gen)):
        X, y = val_gen[i]
        # Use more efficient prediction with batching
        preds = model.predict(X, batch_size=batch_size, verbose=0)
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
    
    # Run CNN approach with parallel processing
    model, history, metrics = visual_cnn_approach(
        data,
        window_size=60,
        prediction_horizon=1,
        img_size=(128, 128),
        batch_size=64,  # Increased batch size for better GPU utilization
        epochs=20,
        num_workers=8,  # Adjust based on your CPU cores
        use_prefetch=True,
        prefetch_limit=2000  # Limit prefetching to save memory
    )
    
    print("\nParallelized Visual CNN Approach Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])