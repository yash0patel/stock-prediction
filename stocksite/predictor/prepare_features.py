import sys
import os
import warnings
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import date, timedelta

# --- Django environment ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksite.settings')
import django
django.setup()

from predictor.models import Stock, News

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# --- Constants ---
YEARS_LOOKBACK = 2.5

today = date.today()
END_DATE = today
START_DATE = END_DATE - timedelta(days=int(365 * YEARS_LOOKBACK))

data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)

# --- Helper Functions ---

def download_data(ticker):
    """Download price data from yfinance"""
    try:
        logger.info(f"Downloading data for {ticker}")
        df = yf.download(
            ticker,
            start=START_DATE.strftime('%Y-%m-%d'),
            end=(END_DATE + timedelta(days=1)).strftime('%Y-%m-%d'),
            progress=False
        )
        if df.empty:
            logger.warning(f"No price data for {ticker}")
            return None
            
        # Handle MultiIndex columns (when multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
            
        # Reset index to get Date as column
        df.reset_index(inplace=True)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in downloaded data: {missing_cols}")
            return None
            
        logger.info(f"Downloaded {len(df)} rows of data for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None

def merge_sentiment(df, ticker):
    """Merge sentiment data with price data"""
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    news_qs = News.objects.filter(stock__ticker=ticker)
    
    if not news_qs.exists():
        logger.info(f"No news data found for {ticker}, using zero sentiment")
        df['DailySentiment'] = 0.0
        df['SentimentCount'] = 0
        df['SentimentVolatility'] = 0.0
        df['SentimentMA_3'] = 0.0
        df['SentimentMA_7'] = 0.0
        return df
        
    try:
        news_df = pd.DataFrame.from_records(news_qs.values('published_date', 'sentiment'))
        news_df['Date'] = pd.to_datetime(news_df['published_date']).dt.date
        agg = news_df.groupby('Date').sentiment.agg(['mean','count','std']).fillna(0)
        agg.columns = ['DailySentiment','SentimentCount','SentimentVolatility']
        df = df.merge(agg.reset_index(), on='Date', how='left').fillna(0)
        
        # Add sentiment moving averages
        df['SentimentMA_3'] = df['DailySentiment'].rolling(3, min_periods=1).mean()
        df['SentimentMA_7'] = df['DailySentiment'].rolling(7, min_periods=1).mean()
        
        logger.info(f"Merged sentiment data for {ticker}")
        return df
        
    except Exception as e:
        logger.warning(f"Error merging sentiment for {ticker}: {e}")
        df['DailySentiment'] = 0.0
        df['SentimentCount'] = 0
        df['SentimentVolatility'] = 0.0
        df['SentimentMA_3'] = 0.0
        df['SentimentMA_7'] = 0.0
        return df

def add_technical_indicators(df):
    """Add technical analysis indicators"""
    try:
        df = df.sort_values('Date').copy()
        df.set_index('Date', inplace=True)
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
        df['BB_Up'] = df['BB_Mid'] + 2 * bb_std
        df['BB_Low'] = df['BB_Mid'] - 2 * bb_std
        
        # Returns and Volatility
        df['DailyReturn'] = df['Close'].pct_change().fillna(0)
        df['Volatility20'] = df['DailyReturn'].rolling(window=20, min_periods=1).std().fillna(0)
        
        # Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        
        # High-Low indicators
        df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Range'] = df['High'] - df['Low']
        
        df.reset_index(inplace=True)
        logger.info("Added technical indicators")
        return df
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return df

def add_lag_features(df, lags=[1, 2, 3, 5]):
    """Add lagged features for time series modeling"""
    try:
        df = df.sort_values('Date').copy()
        
        for lag in lags:
            # Price lags
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['DailyReturn'].shift(lag)
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            
        # Fill missing values with forward fill for initial rows
        lag_columns = [col for col in df.columns if 'Lag_' in col]
        df[lag_columns] = df[lag_columns].fillna(method='ffill').fillna(0)
        
        logger.info(f"Added lag features for lags: {lags}")
        return df
        
    except Exception as e:
        logger.error(f"Error adding lag features: {e}")
        return df

def create_target(df, threshold=0.01):
    """Create target variable for prediction"""
    try:
        df = df.sort_values('Date').copy()
        
        # Next day's closing price
        df['NextClose'] = df['Close'].shift(-1)
        df['ReturnNext'] = (df['NextClose'] - df['Close']) / df['Close']
        
        def classify_movement(return_val):
            if pd.isna(return_val):
                return None
            if return_val >= threshold:
                return 'Up'
            elif return_val <= -threshold:
                return 'Down'
            else:
                return 'Neutral'
        
        df['Movement'] = df['ReturnNext'].apply(classify_movement)
        
        # Count distribution
        movement_counts = df['Movement'].value_counts()
        logger.info(f"Target distribution: {dict(movement_counts)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating target: {e}")
        return df

def clean_data(df, min_rows=50):
    """Clean and validate the final dataset"""
    try:
        original_len = len(df)
        
        # Remove rows with missing target
        df = df.dropna(subset=['Movement'])
        
        # Forward fill remaining missing values, then drop any remaining NaN
        df = df.fillna(method='ffill').fillna(method='bfill').dropna()
        
        final_len = len(df)
        logger.info(f"Data cleaning: {original_len} → {final_len} rows")
        
        if final_len < min_rows:
            logger.warning(f"Insufficient data after cleaning: {final_len} rows (minimum: {min_rows})")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        return None

def process_stock(stock):
    """Main function to process a single stock"""
    ticker = stock.ticker
    logger.info(f"=" * 50)
    logger.info(f"Processing {ticker}")
    
    try:
        # Step 1: Download fresh price data
        df_price = download_data(ticker)
        if df_price is None:
            logger.error(f"Failed to download data for {ticker}")
            return False
        
        # Step 2: Initialize or load existing CSV
        csv_path = os.path.join(data_dir, f"{ticker}_features.csv")
        
        if os.path.exists(csv_path):
            logger.info(f"Found existing CSV for {ticker}, updating...")
            try:
                df_existing = pd.read_csv(csv_path)
                df_existing['Date'] = pd.to_datetime(df_existing['Date'], errors='coerce')
                
                # Merge with new data, removing duplicates
                df_price['Date'] = pd.to_datetime(df_price['Date'])
                df = pd.concat([df_existing, df_price]).drop_duplicates(subset=['Date'], keep='last')
                df = df.sort_values('Date').reset_index(drop=True)
                
            except Exception as e:
                logger.warning(f"Error loading existing CSV: {e}, creating new one")
                df = df_price.copy()
        else:
            logger.info(f"Creating new CSV for {ticker}")
            df = df_price.copy()
        
        # Step 3: Add all features
        logger.info("Adding sentiment data...")
        df = merge_sentiment(df, ticker)
        
        logger.info("Adding technical indicators...")
        df = add_technical_indicators(df)
        
        logger.info("Adding lag features...")
        df = add_lag_features(df)
        
        logger.info("Creating target variable...")
        df = create_target(df)
        
        logger.info("Cleaning data...")
        df = clean_data(df)
        
        if df is None:
            logger.error(f"Data cleaning failed for {ticker}")
            return False
        
        # Step 4: Save processed data
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Successfully saved {len(df)} rows of features for {ticker}")
        
        # Show final feature summary
        feature_cols = [col for col in df.columns if col not in ['Date', 'Movement', 'NextClose', 'ReturnNext']]
        logger.info(f"Generated {len(feature_cols)} features")
        
        return True
        
    except Exception as e:
        logger.error(f"Critical error processing {ticker}: {e}")
        return False

def main():
    """Process all stocks in database"""
    try:
        stocks = Stock.objects.all()
        if not stocks:
            logger.error("No stocks in database. Run fetch_stocks.py first.")
            return
        
        logger.info(f"Found {len(stocks)} stocks to process")
        successful = 0
        
        for stock in stocks:
            if process_stock(stock):
                successful += 1
        
        logger.info(f"Successfully processed {successful}/{len(stocks)} stocks")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()