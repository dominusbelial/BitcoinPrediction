"""
Data Fixer Script

This script cleans and fixes the Bitcoin historical data CSV file,
ensuring all price columns are in numeric format.
"""

import pandas as pd
import os

def fix_bitcoin_data(input_file='data/bitcoin_historical.csv', output_file=None):
    """
    Fix Bitcoin historical data by converting string columns to numeric
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file. If None, will overwrite input file
    
    Returns:
    pandas.DataFrame: Fixed DataFrame
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Loading data from {input_file}...")
    try:
        # Try to load with automatic index parsing
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying alternative loading method...")
        df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    
    # Check for problematic columns
    print("Checking data types...")
    for col in df.columns:
        print(f"Column '{col}': {df[col].dtype}")
        if df[col].dtype == 'object' and col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            non_numeric_values = df[pd.to_numeric(df[col], errors='coerce').isna()][col].unique()
            if len(non_numeric_values) > 0:
                print(f"  WARNING: Found non-numeric values in '{col}': {non_numeric_values[:5]}")
    
    # Convert price columns to numeric
    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in price_columns:
        if col in df.columns:
            print(f"Converting '{col}' to numeric...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    old_shape = df.shape
    df = df.dropna()
    print(f"Dropped {old_shape[0] - df.shape[0]} rows with NaN values")
    
    # Ensure index is datetime
    if df.index.dtype != 'datetime64[ns]':
        try:
            print("Converting index to datetime...")
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"Could not convert index to datetime: {e}")
    
    # Save fixed data
    print(f"Saving fixed data to {output_file}...")
    df.to_csv(output_file)
    print(f"Fixed data shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Fix the data
    fixed_df = fix_bitcoin_data()
    
    print("\nData cleaning complete!")
    print("You can now run your LSTM and Visual CNN approaches with the fixed data.")