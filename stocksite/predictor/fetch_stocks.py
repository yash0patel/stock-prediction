import sys
import os
import pandas as pd
import joblib

# --- Ensure project root is in sys.path so imports always work ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksite.settings')
import django
django.setup()

from predictor.models import Stock

def clean_symbol(symbol):
    """
    Normalize user input symbol:
    - Strip whitespace
    - Uppercase
    - Add '.NS' suffix if missing for NSE stocks
    """
    symbol = symbol.strip().upper()
    if not symbol.endswith('.NS'):
        symbol += '.NS'
    return symbol

def print_clean(message):
    """
    Print without unwanted characters/symbols.
    """
    print(message)

def handle_symbol(symbol):
    """
    Checks if symbol exists and features/models previously generated.
    If yes, loads and predicts.
    Else, runs pipeline to generate features, train, and predict.
    """
    symbol = clean_symbol(symbol)
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    features_path = os.path.join(data_dir, f"{symbol}_features.csv")
    model_path = os.path.join(models_dir, f"{symbol}_model.pkl")
    
    # Check if symbol exists in DB
    stock = Stock.objects.filter(ticker=symbol).first()
    
    if stock and os.path.exists(features_path) and os.path.exists(model_path):
        print_clean(f"Stock '{symbol}' found with existing features and model.")
        try:
            df = pd.read_csv(features_path)
            model = joblib.load(model_path)
            drop_cols = ['Date', 'Movement', 'Future_Close', 'Future_Return']
            input_df = df.tail(1).drop(columns=[c for c in drop_cols if c in df.columns])
            pred = model.predict(input_df)[0]
            print_clean(f"Prediction for {symbol}: {pred}")
        except Exception as e:
            print_clean(f"Error loading data/model or predicting for '{symbol}': {str(e)}")
    else:
        print_clean(f"Processing new stock symbol: '{symbol}'. This may take some time...")
        try:
            if not stock:
                Stock.objects.create(ticker=symbol, name=symbol)
                print_clean(f"Added '{symbol}' to stock database.")
            # Run feature engineering for only this symbol
            os.system(f"python predictor/prepare_features.py --symbol {symbol}")
            # Run model training for only this symbol
            os.system(f"python predictor/train_model.py --symbol {symbol}")
            # Check again and predict
            if os.path.exists(features_path) and os.path.exists(model_path):
                df = pd.read_csv(features_path)
                model = joblib.load(model_path)
                drop_cols = ['Date', 'Movement', 'Future_Close', 'Future_Return']
                input_df = df.tail(1).drop(columns=[c for c in drop_cols if c in df.columns])
                pred = model.predict(input_df)[0]
                print_clean(f"Prediction for {symbol}: {pred}")
            else:
                print_clean(f"Could not find generated features or model for '{symbol}' after training.")
        except Exception as e:
            print_clean(f"Critical error during processing '{symbol}': {str(e)}")

if __name__ == "__main__":
    user_input = input("Enter NSE Indian stock symbol (e.g., TCS.NS): ")
    handle_symbol(user_input)