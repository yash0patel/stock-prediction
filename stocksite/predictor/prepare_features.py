import sys
import os
import warnings
import logging
from datetime import date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

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

# Always use yesterday as the last full trading day to avoid incomplete data
today = date.today()
END_DATE = today - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=int(365 * YEARS_LOOKBACK))

OUTPUT_DIR = os.path.join(project_root, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def download_data(ticker):
    """Download historical data for ticker, handling MultiIndex columns."""
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
        # Handle MultiIndex (yfinance sometimes returns)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None

def merge_sentiment(df, ticker):
    """Merge daily sentiment stats into price DataFrame."""
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    news_qs = News.objects.filter(stock__ticker=ticker)
    if not news_qs.exists():
        df['DailySentiment'] = 0.0
        df['SentimentCount'] = 0
        df['SentimentVolatility'] = 0.0
        return df

    news_df = pd.DataFrame.from_records(
        news_qs.values('published_date', 'sentiment')
    )
    news_df['Date'] = pd.to_datetime(news_df['published_date']).dt.date
    agg = news_df.groupby('Date').sentiment.agg(['mean','count','std']).fillna(0)
    agg.columns = ['DailySentiment','SentimentCount','SentimentVolatility']
    agg = agg.reset_index()
    df = df.merge(agg, on='Date', how='left')
    df[['DailySentiment','SentimentCount','SentimentVolatility']] = \
        df[['DailySentiment','SentimentCount','SentimentVolatility']].fillna(0)
    df['SentimentMA_3'] = df['DailySentiment'].rolling(3).mean().fillna(0)
    df['SentimentMA_7'] = df['DailySentiment'].rolling(7).mean().fillna(0)
    return df

def add_technical_indicators(df):
    """Compute key technical indicators safely, even with limited data."""
    try:
        df.set_index('Date', inplace=True)
        for w in [5, 10, 20, 50]:
            df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_Up'] = df['BB_Mid'] + 2 * std
        df['BB_Low'] = df['BB_Mid'] - 2 * std
        df['DailyReturn'] = df['Close'].pct_change()
        df['Volatility20'] = df['DailyReturn'].rolling(20).std()
        df.reset_index(inplace=True)
    except Exception as e:
        logger.warning(f"Technical indicator error: {e}")
        df.reset_index(inplace=True)
    return df

def add_lag_features(df, lags=[1,2,3,5]):
    """Add lagged versions of select features."""
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['DailyReturn'].shift(lag)
        df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
    return df

def create_target(df, threshold=0.0):
    """Create 'Movement' target: Up, Down, or Neutral."""
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
    """Clean NaNs and ensure sufficient rows."""
    original = len(df)
    df = df.dropna(subset=['Movement'])
    df = df.fillna(method='ffill').dropna()
    if len(df) < min_rows:
        logger.warning(f"Only {len(df)} rows after cleaning; minimum required is {min_rows}")
        return None
    logger.info(f"Cleaned data: {original} → {len(df)} rows")
    return df

def process_stock(stock):
    ticker = stock.ticker
    logger.info(f"Processing {ticker}")
    df = download_data(ticker)
    if df is None:
        return
    df = merge_sentiment(df, ticker)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = create_target(df, threshold=0.0)
    df = clean_data(df)
    if df is None:
        return
    out_file = os.path.join(OUTPUT_DIR, f"{ticker}_features.csv")
    df.to_csv(out_file, index=False)
    logger.info(f"Saved features to {out_file}")

def main():
    stocks = Stock.objects.all()
    if not stocks:
        logger.error("No stocks in database. Run fetch_stocks first.")
        return
    for stock in stocks:
        try:
            process_stock(stock)
        except Exception as e:
            logger.error(f"Failed to process {stock.ticker}: {e}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logger.info(f"Feature engineering from {START_DATE} to {END_DATE}")
    main()
    logger.info("Feature engineering complete.")
 