import sys
import os
import pandas as pd
import joblib
import yfinance as yf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

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
    Normalize and validate Indian stock symbol:
    - Strip whitespace and uppercase
    - Try NSE (.NS) first, then BSE (.BO)
    - Validate data exists via yfinance
    - Return valid symbol or raise error
    """
    symbol = symbol.strip().upper()
    
    # Remove any existing suffix to start fresh
    base_symbol = symbol.split('.')[0]
    
    # Try NSE first (.NS), then BSE (.BO)
    for exchange in ['.NS', '.BO']:
        test_symbol = base_symbol + exchange
        try:
            logger.info(f"Testing symbol: {test_symbol}")
            ticker = yf.Ticker(test_symbol)
            # Test with small data fetch
            data = ticker.history(period='5d')
            if not data.empty and len(data) > 0:
                logger.info(f"✓ Found valid data for {test_symbol}")
                return test_symbol
        except Exception as e:
            logger.warning(f"✗ No data for {test_symbol}: {str(e)}")
            continue
    
    # If both fail, raise error
    raise ValueError(f"Stock symbol '{symbol}' not found on NSE (.NS) or BSE (.BO). Please check the symbol.")

def print_clean(message):
    """Print clean formatted messages"""
    print(f"[STOCK-PREDICTOR] {message}")

def validate_data_files(symbol, features_path, model_path):
    """Validate that data files exist and are readable"""
    if not os.path.exists(features_path):
        return False, f"Features file missing: {features_path}"
    
    if not os.path.exists(model_path):
        return False, f"Model file missing: {model_path}"
    
    try:
        # Test read features file
        df = pd.read_csv(features_path)
        if df.empty:
            return False, "Features file is empty"
        
        # Test load model file
        model_obj = joblib.load(model_path)
        return True, "Files validated successfully"
        
    except Exception as e:
        return False, f"File validation error: {str(e)}"

def make_prediction(symbol, features_path, model_path):
    """Make prediction using existing model and features"""
    try:
        # Load features
        df = pd.read_csv(features_path)
        if df.empty:
            raise ValueError("No feature data available")
        
        # Load model (handle both dict and direct model formats)
        model_obj = joblib.load(model_path)
        if isinstance(model_obj, dict):
            model = model_obj.get('model')
            scaler = model_obj.get('scaler')
            label_encoder = model_obj.get('le')
        else:
            model = model_obj
            scaler = None
            label_encoder = None
        
        # Prepare input data (use last row)
        latest_data = df.tail(1).copy()
        
        # Drop target and identifier columns
        drop_cols = ['Date', 'Movement', 'Future_Close', 'Future_Return', 'NextClose', 'ReturnNext']
        feature_cols = [col for col in latest_data.columns if col not in drop_cols]
        X = latest_data[feature_cols]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale if scaler available
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Decode if label encoder available
        if label_encoder is not None:
            try:
                prediction = label_encoder.inverse_transform([prediction])[0]
            except:
                pass
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_scaled)[0]
                confidence = max(proba)
            except:
                pass
        
        return prediction, confidence
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def run_data_pipeline(symbol):
    """Run the complete data pipeline for a symbol"""
    try:
        print_clean("Step 1: Running feature generation...")
        
        # Import and run feature generation
        sys.path.append(os.path.join(project_root, 'predictor'))
        from prepare_features import process_stock
        
        # Get or create stock object
        stock, created = Stock.objects.get_or_create(ticker=symbol, defaults={'name': symbol})
        
        # Process features
        process_stock(stock)
        
        print_clean("Step 2: Training model...")
        
        # Import and run model training
        from train_model import train_one
        
        features_path = os.path.join(project_root, 'data', f"{symbol}_features.csv")
        if os.path.exists(features_path):
            train_one(features_path)
            print_clean("✓ Pipeline completed successfully")
            return True
        else:
            print_clean("✗ Feature generation failed")
            return False
            
    except Exception as e:
        print_clean(f"✗ Pipeline failed: {str(e)}")
        return False

def handle_symbol(symbol):
    """
    Main function to handle stock symbol processing and prediction
    """
    try:
        # Step 1: Validate and clean symbol
        print_clean(f"Validating symbol: {symbol}")
        symbol = clean_symbol(symbol)
        print_clean(f"Using validated symbol: {symbol}")
        
    except ValueError as e:
        print_clean(f"ERROR: {str(e)}")
        print_clean("Available Indian stock examples: TCS, RELIANCE, INFY, HDFC, SBIN")
        return
    
    # Step 2: Setup paths
    base_dir = project_root
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    features_path = os.path.join(data_dir, f"{symbol}_features.csv")
    model_path = os.path.join(models_dir, f"{symbol}_best_model.pkl")
    
    # Step 3: Check if data exists and is valid
    is_valid, validation_msg = validate_data_files(symbol, features_path, model_path)
    
    if is_valid:
        print_clean(f"Found existing data for {symbol}")
        try:
            prediction, confidence = make_prediction(symbol, features_path, model_path)
            print_clean("="*50)
            print_clean(f"PREDICTION FOR {symbol}")
            print_clean(f"Result: {prediction}")
            if confidence:
                print_clean(f"Confidence: {confidence:.2%}")
            print_clean("="*50)
            
        except Exception as e:
            print_clean(f"Prediction error: {str(e)}")
            print_clean("Regenerating data...")
            if run_data_pipeline(symbol):
                try:
                    prediction, confidence = make_prediction(symbol, features_path, model_path)
                    print_clean("="*50)
                    print_clean(f"PREDICTION FOR {symbol}")
                    print_clean(f"Result: {prediction}")
                    if confidence:
                        print_clean(f"Confidence: {confidence:.2%}")
                    print_clean("="*50)
                except Exception as e2:
                    print_clean(f"Final prediction error: {str(e2)}")
    else:
        print_clean(f"No existing data found: {validation_msg}")
        print_clean(f"Generating new data for {symbol}... This may take 2-3 minutes.")
        
        if run_data_pipeline(symbol):
            try:
                prediction, confidence = make_prediction(symbol, features_path, model_path)
                print_clean("="*50)
                print_clean(f"PREDICTION FOR {symbol}")
                print_clean(f"Result: {prediction}")
                if confidence:
                    print_clean(f"Confidence: {confidence:.2%}")
                print_clean("="*50)
            except Exception as e:
                print_clean(f"Final prediction error: {str(e)}")
        else:
            print_clean("Failed to generate data and model")

if __name__ == "__main__":
    print_clean("Indian Stock Price Movement Predictor")
    print_clean("Supports NSE and BSE stocks")
    print("="*50)
    
    user_input = input("Enter Indian stock symbol (TCS, RELIANCE, INFY, etc.): ").strip()
    
    if user_input:
        handle_symbol(user_input)
    else:
        print_clean("No symbol entered. Exiting.")