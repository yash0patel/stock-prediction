import sys
import os
import warnings
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import date, timedelta
from groq import Groq

# Import the AI patching script
from patch_missing_with_ai import patch_csv_with_ai

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
    """Download price data"""
    try:
        df = yf.download(
            ticker,
            start=START_DATE.strftime('%Y-%m-%d'),
            end=(END_DATE + timedelta(days=1)).strftime('%Y-%m-%d'),
            progress=False
        )
        if df.empty:
            logger.warning(f"No price data for {ticker}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None


def merge_sentiment(df, ticker):
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    news_qs = News.objects.filter(stock__ticker=ticker)
    if not news_qs.exists():
        df['DailySentiment'] = 0.0
        df['SentimentCount'] = 0
        df['SentimentVolatility'] = 0.0
        return df
    news_df = pd.DataFrame.from_records(news_qs.values('published_date', 'sentiment'))
    news_df['Date'] = pd.to_datetime(news_df['published_date']).dt.date
    agg = news_df.groupby('Date').sentiment.agg(['mean','count','std']).fillna(0)
    agg.columns = ['DailySentiment','SentimentCount','SentimentVolatility']
    df = df.merge(agg.reset_index(), on='Date', how='left').fillna(0)
    df['SentimentMA_3'] = df['DailySentiment'].rolling(3, min_periods=1).mean()
    df['SentimentMA_7'] = df['DailySentiment'].rolling(7, min_periods=1).mean()
    return df


def add_technical_indicators(df):
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    for w in [5, 10, 20, 50]:
        df[f'SMA_{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_Mid'] = df['Close'].rolling(window=20, min_periods=1).mean()
    std = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Up'] = df['BB_Mid'] + 2 * std
    df['BB_Low'] = df['BB_Mid'] - 2 * std
    df['DailyReturn'] = df['Close'].pct_change().fillna(0)
    df['Volatility20'] = df['DailyReturn'].rolling(window=20, min_periods=1).std().fillna(0)
    df.reset_index(inplace=True)
    return df


def add_lag_features(df, lags=[1,2,3,5]):
    df = df.sort_values('Date')
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag).fillna(method='ffill')
        df[f'Return_Lag_{lag}'] = df['DailyReturn'].shift(lag).fillna(0)
        df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag).fillna(method='ffill')
    return df


def create_target(df, threshold=0.0):
    df = df.sort_values('Date')
    df['NextClose'] = df['Close'].shift(-1)
    df['ReturnNext'] = (df['NextClose'] - df['Close']) / df['Close']
    def label(r):
        if pd.isna(r): return None
        if r >= threshold: return 'Up'
        if r <= -threshold: return 'Down'
        return 'Neutral'
    df['Movement'] = df['ReturnNext'].apply(label)
    return df


def clean_data(df, min_rows=50):
    original = len(df)
    df = df.dropna(subset=['Movement']).fillna(method='ffill').dropna()
    if len(df) < min_rows:
        logger.warning(f"Only {len(df)} rows after cleaning; minimum required is {min_rows}")
        return None
    logger.info(f"Cleaned data: {original} → {len(df)} rows")
    return df

# --- Main Processing ---

def process_stock(stock):
    ticker = stock.ticker
    logger.info(f"Processing {ticker}")

    # Step 1: Download raw price data
    df_price = download_data(ticker)
    if df_price is None:
        return

    # Step 2: Load existing CSV, parse dates
    csv_path = os.path.join(data_dir, f"{ticker}_features.csv")
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isnull().any():
        logger.error("Invalid dates in CSV; cannot parse 'Date'")
        return

    # Step 3: Compute and overwrite full features
    df = merge_sentiment(df, ticker)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = create_target(df)
    df = clean_data(df)
    if df is None:
        return
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved full features for {ticker}")

    # Step 4: Finally append any AI-missing rows
    patch_csv_with_ai(csv_path, ticker)
    logger.info(f"Appended AI-missing rows for {ticker}")


def main():
    stocks = Stock.objects.all()
    if not stocks:
        logger.error("No stocks in database. Run fetch_stocks first.")
        return
    for s in stocks:
        process_stock(s)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()