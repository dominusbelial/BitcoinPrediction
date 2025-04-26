# Bitcoin Price Prediction Toolkit

This toolkit implements multiple approaches for predicting Bitcoin price movements. It includes traditional statistical methods, machine learning algorithms, and a novel visual CNN approach that uses candlestick chart images.

## Implemented Models

1. **Visual CNN Approach**: Converts candlestick charts to images and uses convolutional neural networks to predict price movements
2. **LSTM Approach**: Uses Long Short-Term Memory networks to capture temporal patterns in the time series data
3. **ARIMA Approach**: Applies traditional statistical time series modeling
4. **XGBoost Approach**: Employs gradient boosting with technical indicators
5. **Random Forest Approach**: Utilizes ensemble learning with technical indicators and lagged features

## Setup and Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/bitcoin-prediction-toolkit.git
cd bitcoin-prediction-toolkit
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Models

Each model can be run independently:

```bash
# Run the Visual CNN approach
python visual_cnn/visual_cnn_approach.py

# Run the LSTM approach
python lstm/lstm_approach.py

# Run the ARIMA approach
python arima/arima_approach.py

# Run the XGBoost approach
python xgboost/xgboost_approach.py

# Run the Random Forest approach
python random_forest/random_forest_approach.py
```

### Comparing All Models

To run all models and compare their performance:

```bash
python compare_models.py
```

This will:
1. Fetch Bitcoin historical data (if not already downloaded)
2. Run all five prediction approaches
3. Generate comparison charts and tables
4. Save the results to the 'results' directory

## Project Structure

```
bitcoin-prediction-toolkit/
├── btc_prediction_toolkit.py   # Common utility functions
├── compare_models.py           # Script to run and compare all models
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── data/                       # Directory for data files
├── models/                     # Directory for saved models
├── results/                    # Directory for results and visualizations
├── visual_cnn/                 # Visual CNN approach
├── lstm/                       # LSTM approach
├── arima/                      # ARIMA approach
├── xgboost/                    # XGBoost approach
└── random_forest/              # Random Forest approach
```

## Customization

You can customize the models by modifying their parameters:

- **Window Size**: Number of previous days to consider (affects all models)
- **Prediction Horizon**: How many days ahead to predict (default is 1 day)
- **Technical Indicators**: Add or remove indicators in `btc_prediction_toolkit.py`
- **Model Hyperparameters**: Tune parameters in each model's script

## Results

Results will be saved in the 'results' directory, including:
- Model comparison metrics (accuracy, precision, recall, F1 score)
- Prediction visualizations for each model
- Feature importance charts for applicable models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data is fetched from Yahoo Finance using the yfinance library
- Code is organized for educational purposes and can be extended for research or practical use