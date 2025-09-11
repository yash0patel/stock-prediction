# patch_missing_with_ai
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import date, timedelta
import json
from groq import Groq

# ---- CONFIG ----
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(project_root, 'data')
TICKER = 'TCS.NS'   # Change for other tickers
csv_path = os.path.join(DATA_DIR, f"{TICKER}_features.csv")

CONFIG_FILE = os.path.join(project_root, "config.json")
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
        GROQ_API_KEY = config.get("GROQ_API_KEY")
else:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

FEATURES = [
    "Date", "Close", "High", "Low", "Open", "Volume",
    "DailySentiment", "SentimentCount", "SentimentVolatility",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26", "MACD", "MACD_Signal", "RSI",
    "BB_Mid", "BB_Up", "BB_Low", "DailyReturn", "Volatility20",
    "Close_Lag_1", "Return_Lag_1", "RSI_Lag_1",
    "Close_Lag_2", "Return_Lag_2", "RSI_Lag_2",
    "Close_Lag_3", "Return_Lag_3", "RSI_Lag_3",
    "Close_Lag_5", "Return_Lag_5", "RSI_Lag_5",
    "NextClose", "ReturnNext", "Movement"
]

def get_existing_dates(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values('Date')
    return df, set(dt.date() for dt in df['Date'])

def find_missing_dates(existing_dates, start_date, end_date):
    all_days = pd.bdate_range(start=start_date, end=end_date)
    return [d.date() for d in all_days if d.date() not in existing_dates]

def generate_groq_row(ticker, missing_date, last_close, last_volume, recent_trend, context_rows=None):
    # Optionally, send last N rows in prompt for better accuracy
    if context_rows is not None and not context_rows.empty:
        context_str = context_rows[FEATURES].to_json(orient='records', date_format='iso')
    else:
        context_str = "[]"

    feature_str = ", ".join(FEATURES)
    prompt = f"""
You are a financial data engineer. Given the prior 5 days of feature rows (JSON below), and the context, generate a new JSON stock feature row for {ticker.replace('.NS',' Indian stock')} for {missing_date}.

Previous 5 rows (JSON): {context_str}

- Output feature keys: [{feature_str}]
- Compute all technical and lag features as per standard finance/statistics/ta-lib meaning; SMA/EMA, RSI, MACD, percent and lag fields.
- Use 0 for all sentiment fields (sentiment, count, volatility) if no news.
- For 'Movement', assign "Up" if ReturnNext>0, "Down" if <0, else "Neutral".
- All price changes/volumes must be realistic for Indian large caps (see context; prices max ±5%).
- Output VALID JSON. DO NOT explain!
- Only generate the row for {missing_date}.
"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Return only valid JSON using all columns, no explanation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200, temperature=0.35, top_p=1, stream=False
        )
        ai_response = completion.choices[0].message.content.strip()
        ai_response = ai_response.replace("``````", "").strip()
        obj = json.loads(ai_response)
        # Fallback fill for any missing fields
        for col in FEATURES:
            if col not in obj: obj[col] = 0
        return obj
    except Exception as e:
        logger.warning(f"Groq fallback for {missing_date}: {e}")
        # All-zeros fallback
        fallback = {col: 0 for col in FEATURES}
        fallback['Date'] = str(missing_date)
        fallback['Close'] = fallback['High'] = fallback['Low'] = fallback['Open'] = 3800.0
        fallback['Volume'] = 1000000
        return fallback

def patch_csv_with_ai(csv_path, ticker, lookback=5):
    df, existing_dates = get_existing_dates(csv_path)
    today = date.today()
    last_date_in_csv = max(existing_dates)
    yesterday = today - timedelta(days=1)
    missing_dates = find_missing_dates(existing_dates, last_date_in_csv + timedelta(days=1), yesterday)
    if not missing_dates:
        logger.info("No missing dates, file up to date.")
        return

    logger.info(f"Missing dates: {missing_dates}")

    ai_rows = []
    for d in missing_dates:
        prev_rows = df[df["Date"] < pd.Timestamp(d)].tail(lookback)
        if not prev_rows.empty:
            last_close = prev_rows.iloc[-1]["Close"]
            last_volume = prev_rows.iloc[-1]["Volume"]
            trend = "up" if len(prev_rows) > 1 and prev_rows.iloc[-1]['Close'] > prev_rows.iloc[-2]['Close'] else "down"
        else:
            last_close = 3800.0
            last_volume = 1000000
            trend = "neutral"
        ai_row = generate_groq_row(ticker, d, last_close, last_volume, trend, prev_rows)
        ai_rows.append(ai_row)

    # Convert into DataFrame, ensure all required columns/order
    ai_df = pd.DataFrame(ai_rows)
    ai_df['Date'] = pd.to_datetime(ai_df['Date'])
    for col in FEATURES:
        if col not in ai_df:
            ai_df[col] = 0
    ai_df = ai_df[FEATURES]
    ai_df = ai_df.sort_values("Date")
    ai_df.to_csv(csv_path, mode='a', index=False, header=False)
    logger.info(f"Patched {len(ai_df)} AI feature rows into {csv_path}")

if __name__ == "__main__":
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY missing: Set in config.json or as environment variable.")
        sys.exit(1)
    patch_csv_with_ai(csv_path, TICKER)
