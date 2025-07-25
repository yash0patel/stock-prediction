import sys
import os
import time
import logging
import feedparser
import pandas as pd
import numpy as np

# Ensure project root is in sys.path so Django and predictors always import correctly, even if run from anywhere
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksite.settings')
import django
django.setup()

from predictor.models import News, Stock

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

RSS_URL = 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms'
FETCH_LIMIT = 50  # number of headlines to fetch

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.pos_kw = {
            'profit', 'growth', 'surge', 'rally', 'beat', 'upgrade',
            'outperform', 'bullish', 'strong', 'record', 'breakthrough',
            'expansion', 'robust', 'solid', 'increase', 'gain', 'optimistic',
            'recovery', 'surpass', 'acquisition', 'merger', 'dividend',
            'buyback', 'ipo', 'buy', 'uptrend', 'raise', 'approval', 'order',
            'deal', 'agreement', 'highest', 'stable', 'resilient'
        }
        self.neg_kw = {
            'decline', 'fall', 'miss', 'downgrade', 'risk', 'crash',
            'bearish', 'slump', 'drop', 'loss', 'plunge', 'volatility',
            'concern', 'slowdown', 'deficit', 'inflation', 'weakness',
            'uncertain', 'selloff', 'underperform', 'negative', 'warn',
            'cut', 'fraud', 'lawsuit', 'default', 'lower', 'strike', 'probe'
        }
        self.sentiment_thresholds = {
            'positive': 0.15,
            'negative': -0.15,
            'neutral_lower': -0.15,
            'neutral_upper': 0.15
        }

    def analyze(self, text):
        # Ignore empty input
        if not text or pd.isna(text):
            return 0.0
        text_l = text.lower()
        vader_score = self.vader.polarity_scores(text)['compound']
        pos_boost = sum(0.1 for kw in self.pos_kw if kw in text_l)
        neg_boost = -sum(0.1 for kw in self.neg_kw if kw in text_l)
        final_score = vader_score + pos_boost + neg_boost
        if TEXTBLOB_AVAILABLE:
            tb_score = TextBlob(text).sentiment.polarity
            final_score = (0.7 * final_score) + (0.3 * tb_score)
        return max(-1.0, min(1.0, final_score))

    def classify(self, score):
        if score >= self.sentiment_thresholds['positive']:
            return 'Positive'
        if score <= self.sentiment_thresholds['negative']:
            return 'Negative'
        return 'Neutral'

def fetch_rss_headlines(url, limit=50):
    feed = feedparser.parse(url)
    news = []
    for e in feed.entries[:limit]:
        published = e.get('published', e.get('updated', ''))
        title = e.get('title', '').strip()
        if title and published:
            news.append((title, published))
    return news

def update_news_entries():
    analyzer = SentimentAnalyzer()
    fetched = fetch_rss_headlines(RSS_URL, FETCH_LIMIT)
    count_new = 0
    stocks = list(Stock.objects.all())
    for title, pub in fetched:
        try:
            dt = pd.to_datetime(pub)
        except Exception:
            continue
        for stock in stocks:
            # Uses company symbol as substring, can be customized for more accuracy
            sym = stock.ticker.split('.')[0]
            if sym and sym in title:
                exists = News.objects.filter(stock=stock, headline=title).exists()
                if not exists:
                    news = News(stock=stock, headline=title, published_date=dt)
                    news.sentiment = analyzer.analyze(title)
                    news.save()
                    count_new += 1
    logger.info(f"Added {count_new} new news entries.")

def aggregate_daily_sentiment():
    """Summarize daily sentiment for reporting or reuse by feature scripts."""
    qs = News.objects.values('published_date', 'sentiment')
    df = pd.DataFrame.from_records(qs)
    if df.empty:
        logger.info("No news records to aggregate.")
        return
    df['date'] = pd.to_datetime(df['published_date']).dt.date
    agg = df.groupby('date').sentiment.agg(['mean', 'count', 'std']).fillna(0)
    agg = agg.rename(columns={
        'mean': 'DailyMeanSentiment',
        'count': 'DailySentimentCount',
        'std': 'DailySentimentVolatility'
    })
    logger.info("Aggregated daily sentiment statistics for later modeling.")

if __name__ == "__main__":
    logger.info("Starting sentiment analysis...")
    update_news_entries()
    aggregate_daily_sentiment()
    logger.info("Sentiment analysis complete.")
