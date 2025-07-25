# analyze_sentiment.py - Advanced Multi-Method Sentiment Analysis
import sys
import os
import time
import logging
from datetime import datetime

# Django setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stocksite.settings")
import django
django.setup()

# Sentiment analysis imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available - using VADER only")

from predictor.models import News, Stock
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    """Advanced sentiment analyzer with multiple methods and Indian market adaptation[3][6][12]"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Indian market-specific keywords for sentiment enhancement[43][46]
        self.positive_keywords = {
            'quarterly', 'profit', 'growth', 'expansion', 'merger', 'acquisition',
            'dividend', 'bonus', 'upgrade', 'outperform', 'buy', 'bullish',
            'strong', 'beat', 'exceed', 'record', 'highest', 'surge', 'rally',
            'nifty', 'sensex', 'ipo', 'fii', 'dii'
        }
        
        self.negative_keywords = {
            'loss', 'decline', 'fall', 'crash', 'bear', 'sell', 'downgrade',
            'weak', 'miss', 'disappoint', 'concern', 'risk', 'volatility',
            'correction', 'recession', 'inflation', 'deficit'
        }
        
        # Optimized VADER thresholds for financial news[54][56]
        self.sentiment_thresholds = {
            'positive': 0.15,    # Lowered from default 0.05 for more precision
            'negative': -0.15,   # Lowered from default -0.05 for more precision
            'neutral_lower': -0.15,
            'neutral_upper': 0.15
        }
        
    def analyze_vader_sentiment(self, text):
        """Enhanced VADER sentiment analysis with Indian market context"""
        if not text or pd.isna(text):
            return 0.0
            
        # Clean and prepare text
        text = str(text).lower().strip()
        
        # Get base VADER scores
        scores = self.vader_analyzer.polarity_scores(text)
        base_compound = scores['compound']
        
        # Apply Indian market keyword boosting
        positive_boost = sum(1 for keyword in self.positive_keywords if keyword in text) * 0.1
        negative_boost = sum(1 for keyword in self.negative_keywords if keyword in text) * -0.1
        
        # Calculate enhanced sentiment
        enhanced_sentiment = base_compound + positive_boost + negative_boost
        
        # Apply bounds checking
        enhanced_sentiment = max(-1.0, min(1.0, enhanced_sentiment))
        
        return enhanced_sentiment
    
    def analyze_textblob_sentiment(self, text):
        """TextBlob sentiment analysis as secondary method"""
        if not TEXTBLOB_AVAILABLE or not text or pd.isna(text):
            return 0.0
            
        try:
            blob = TextBlob(str(text))
            # TextBlob returns polarity in [-1, 1] range
            return float(blob.sentiment.polarity)
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return 0.0
    
    def get_ensemble_sentiment(self, text):
        """Ensemble sentiment combining multiple methods for higher accuracy[30][36]"""
        if not text or pd.isna(text):
            return 0.0
        
        # Get sentiment scores from different methods
        vader_score = self.analyze_vader_sentiment(text)
        textblob_score = self.analyze_textblob_sentiment(text) if TEXTBLOB_AVAILABLE else 0.0
        
        # Weighted ensemble (VADER gets higher weight for financial text)
        if TEXTBLOB_AVAILABLE:
            ensemble_score = (0.7 * vader_score) + (0.3 * textblob_score)
        else:
            ensemble_score = vader_score
        
        return float(ensemble_score)
    
    def classify_sentiment(self, score):
        """Classify sentiment using optimized thresholds"""
        if score >= self.sentiment_thresholds['positive']:
            return 'Positive'
        elif score <= self.sentiment_thresholds['negative']:
            return 'Negative'
        else:
            return 'Neutral'

def process_stock_sentiment(stock, analyzer):
    """Process sentiment for a specific stock with error handling"""
    try:
        news_items = News.objects.filter(stock=stock)
        
        if not news_items.exists():
            logger.info(f"📰 No news found for {stock.ticker}")
            return {"processed": 0, "errors": 0}
        
        processed_count = 0
        error_count = 0
        
        for news in news_items:
            try:
                # Get ensemble sentiment score
                sentiment_score = analyzer.get_ensemble_sentiment(news.headline)
                sentiment_class = analyzer.classify_sentiment(sentiment_score)
                
                # Update news record
                news.sentiment = sentiment_score
                news.save()
                
                processed_count += 1
                
                # Log detailed results for monitoring
                logger.debug(f"📊 {stock.ticker}: '{news.headline[:50]}...' → {sentiment_score:.3f} ({sentiment_class})")
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Error processing news ID {news.id}: {str(e)}")
                continue
        
        logger.info(f"✅ {stock.ticker}: Processed {processed_count} headlines, {error_count} errors")
        return {"processed": processed_count, "errors": error_count}
        
    except Exception as e:
        logger.error(f"💥 Critical error processing {stock.ticker}: {str(e)}")
        return {"processed": 0, "errors": 1}

def generate_sentiment_report(results):
    """Generate comprehensive sentiment analysis report"""
    total_processed = sum(r["processed"] for r in results.values())
    total_errors = sum(r["errors"] for r in results.values())
    
    # Calculate sentiment distribution
    all_news = News.objects.exclude(sentiment=0)
    
    if all_news.exists():
        sentiments = [news.sentiment for news in all_news]
        
        positive_count = len([s for s in sentiments if s >= 0.15])
        negative_count = len([s for s in sentiments if s <= -0.15])
        neutral_count = len(sentiments) - positive_count - negative_count
        
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        logger.info("=" * 60)
        logger.info("📈 SENTIMENT ANALYSIS REPORT")
        logger.info("=" * 60)
        logger.info(f"📊 Total headlines processed: {total_processed}")
        logger.info(f"❌ Total errors: {total_errors}")
        logger.info(f"📈 Positive sentiment: {positive_count} ({positive_count/len(sentiments)*100:.1f}%)")
        logger.info(f"📉 Negative sentiment: {negative_count} ({negative_count/len(sentiments)*100:.1f}%)")
        logger.info(f"📊 Neutral sentiment: {neutral_count} ({neutral_count/len(sentiments)*100:.1f}%)")
        logger.info(f"🎯 Average sentiment: {avg_sentiment:.3f}")
        logger.info(f"📏 Sentiment volatility (std): {sentiment_std:.3f}")
        logger.info("=" * 60)

def main():
    """Main sentiment analysis execution"""
    start_time = time.time()
    
    logger.info("🚀 Starting enhanced sentiment analysis...")
    
    try:
        # Initialize advanced sentiment analyzer
        analyzer = EnhancedSentimentAnalyzer()
        
        # Get all stocks with news
        stocks_with_news = Stock.objects.filter(news__isnull=False).distinct()
        
        if not stocks_with_news.exists():
            logger.warning("⚠️ No stocks with news found. Run fetch_stocks.py first.")
            return
        
        logger.info(f"🎯 Processing sentiment for {stocks_with_news.count()} stocks...")
        
        # Process each stock
        results = {}
        for stock in stocks_with_news:
            results[stock.ticker] = process_stock_sentiment(stock, analyzer)
            
            # Rate limiting to avoid overwhelming the system
            time.sleep(0.1)
        
        # Generate comprehensive report
        generate_sentiment_report(results)
        
        execution_time = time.time() - start_time
        logger.info(f"⏱️ Sentiment analysis completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"💥 Critical error in sentiment analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
