# predictor/prepare_features.py
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Django setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stocksite.settings")
import django
django.setup()

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import time

from predictor.models import Stock, News

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

START_DATE = "2023-01-01"
END_DATE   = "2025-01-25"

class AdvancedFeatureEngineer:

    def calculate_technical_indicators(self, df):
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - 100 / (1 + gain / loss)
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(20).std()
        return df

    def add_lag_features(self, df, lags=[1,2,3,5]):
        for lag in lags:
            df[f'Close_Lag_{lag}']  = df['Close'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
        return df

    def merge_sentiment(self, df, stock):
        news_qs = News.objects.filter(stock=stock).values('published_date','sentiment')
        if not news_qs:
            df['DailySentiment'] = 0.0
            df['SentimentCount'] = 0
            return df

        news_df = pd.DataFrame(list(news_qs))
        news_df['Date'] = pd.to_datetime(news_df['published_date']).dt.date
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        agg = (news_df
               .groupby('Date')
               .sentiment
               .agg(['mean','count'])
               .rename(columns={'mean':'DailySentiment','count':'SentimentCount'})
               .reset_index())
        df = df.merge(agg, on='Date', how='left')
        df['DailySentiment'].fillna(0, inplace=True)
        df['SentimentCount'].fillna(0, inplace=True)
        return df

    def create_target(self, df, horizon=1, up_thr=0.02, down_thr=-0.02):
        df['Future_Close']  = df['Close'].shift(-horizon)
        df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']

        def lbl(x):
            if pd.isna(x):    return None
            if x >= up_thr:   return 'Up'
            if x <= down_thr: return 'Down'
            return 'Neutral'

        df['Movement'] = df['Future_Return'].apply(lbl)
        return df

def download_data(ticker):
    df = yf.download(ticker,
                     start=START_DATE,
                     end=END_DATE,
                     progress=False)
    if df.empty:
        logger.warning(f"No data for {ticker}")
        return None

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()[['Date','Open','High','Low','Close','Volume']]
    return df

def process_stock(stock):
    logger.info(f"Processing {stock.ticker}")
    df = download_data(stock.ticker)
    if df is None or len(df) < 50:
        logger.warning(f"Insufficient data for {stock.ticker}")
        return False

    fe = AdvancedFeatureEngineer()
    df = fe.calculate_technical_indicators(df)
    df = fe.add_lag_features(df)
    df = fe.merge_sentiment(df, stock)
    df = fe.create_target(df)

    # Drop any rows where 'Movement' is missing
    df.dropna(subset=['Movement'], inplace=True)

    out_file = f"{stock.ticker}_features.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Saved features to {out_file} (rows: {len(df)})")
    return True

def main():
    start = time.time()
    stocks = Stock.objects.all()
    if not stocks.exists():
        logger.error("No stocks found in DB.")
        return

    success, fail = [], []
    for s in stocks:
        ok = process_stock(s)
        (success if ok else fail).append(s.ticker)
        time.sleep(0.5)

    logger.info(f"Completed in {time.time()-start:.1f}s. Success: {len(success)}, Fail: {len(fail)}")
    if success: logger.info("OK:  " + ", ".join(success))
    if fail:    logger.warning("FAIL:" + ", ".join(fail))

if __name__ == "__main__":
    main()
