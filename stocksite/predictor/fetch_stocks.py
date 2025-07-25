# fetch_stocks.py - Enhanced Indian Stock Data Fetcher
import sys
import os
import time
import logging
from datetime import datetime, timedelta

# Django setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stocksite.settings")
import django
django.setup()

from predictor.models import Stock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_stock_database():
    """Enhanced stock database with high-volume Indian stocks"""
    
    # Premium Indian stocks with high liquidity and reliable data[1][5][17]
    indian_stocks = [
        # Banking & Financial Services
        ("HDFCBANK.NS", "HDFC Bank Limited"),
        ("ICICIBANK.NS", "ICICI Bank Limited"),
        ("SBIN.NS", "State Bank of India"),
        ("KOTAKBANK.NS", "Kotak Mahindra Bank"),
        ("AXISBANK.NS", "Axis Bank Limited"),
        
        # Information Technology
        ("TCS.NS", "Tata Consultancy Services"),
        ("INFY.NS", "Infosys Limited"),
        ("HCLTECH.NS", "HCL Technologies"),
        ("WIPRO.NS", "Wipro Limited"),
        ("TECHM.NS", "Tech Mahindra"),
        
        # Oil", "Reliance Industries"),
        ("ONGC.NS", "Oil & Natural Gas Corporation"),
        ("IOC.NS", "Indian Oil Corporation"),
        ("BPCL.NS", "Bharat Petroleum Corporation"),
        
        # Pharmaceuticals
        ("SUNPHARMA.NS", "Sun Pharmaceutical Industries"),
        ("DRREDDY.NS", "Dr. Reddy's Laboratories"),
        ("CIPLA.NS", "Cipla Limited"),
        ("DIVISLAB.NS", "Divi's Laboratories"),
        
        # Consumer Goods
        ("HINDUNILVR.NS", "Hindustan Unilever"),
        ("ITC.NS", "ITC Limited"),
        ("NESTLEIND.NS", "Nestle India"),
        ("BRITANNIA.NS", "Britannia Industries"),
        
        # Automotive
        ("MARUTI.NS", "Maruti Suzuki India"),
        ("TATAMOTORS.NS", "Tata Motors"),
        ("M&M.NS", "Mahindra & Mahindra"),
        ("BAJAJ-AUTO.NS", "Bajaj Auto"),
        
        # Metals & Mining
        ("TATASTEEL.NS", "Tata Steel"),
        ("HINDALCO.NS", "Hindalco Industries"),
        ("JSWSTEEL.NS", "JSW Steel"),
        ("COALINDIA.NS", "Coal India"),
        
        # Telecom
        ("BHARTIARTL.NS", "Bharti Airtel"),
        ("JIO.NS", "Reliance Jio Infocomm"),
        
        # Cement
        ("ULTRACEMCO.NS", "UltraTech Cement"),
        ("SHREECEM.NS", "Shree Cement"),
        
        # Power & Infrastructure
        ("NTPC.NS", "NTPC Limited"),
        ("POWERGRID.NS", "Power Grid Corporation"),
        ("LT.NS", "Larsen & Toubro"),
        
        # FMCG
        ("ASIANPAINT.NS", "Asian Paints"),
        ("TITAN.NS", "Titan Company")
    ]
    
    success_count = 0
    total_stocks = len(indian_stocks)
    
    logger.info(f"Starting to populate stock database with {total_stocks} Indian stocks...")
    
    for ticker, name in indian_stocks:
        try:
            stock, created = Stock.objects.get_or_create(
                ticker=ticker,
                defaults={'name': name}
            )
            
            if created:
                logger.info(f"✅ Created: {ticker} - {name}")
                success_count += 1
            else:
                logger.info(f"🔄 Exists: {ticker} - {name}")
            
            # Rate limiting to avoid overwhelming Yahoo Finance[23][29]
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"❌ Error processing {ticker}: {str(e)}")
            continue
    
    logger.info(f"✨ Stock database population complete!")
    logger.info(f"📊 Successfully processed: {success_count} new stocks")
    logger.info(f"📈 Total stocks in database: {Stock.objects.count()}")
    
    return success_count

def validate_stock_data():
    """Validate that stocks can fetch data successfully"""
    import yfinance as yf
    
    logger.info("🔍 Validating stock data availability...")
    
    valid_stocks = []
    invalid_stocks = []
    
    for stock in Stock.objects.all()[:5]:  # Test first 5 stocks
        try:
            # Test data fetch with error handling[8][14]
            ticker_obj = yf.Ticker(stock.ticker)
            hist = ticker_obj.history(period="5d")
            
            if not hist.empty and len(hist) > 0:
                valid_stocks.append(stock.ticker)
                logger.info(f"✅ {stock.ticker}: Data available ({len(hist)} days)")
            else:
                invalid_stocks.append(stock.ticker)
                logger.warning(f"⚠️ {stock.ticker}: No data available")
                
        except Exception as e:
            invalid_stocks.append(stock.ticker)
            logger.error(f"❌ {stock.ticker}: Error - {str(e)}")
        
        time.sleep(1)  # Rate limiting
    
    logger.info(f"📈 Validation complete: {len(valid_stocks)} valid, {len(invalid_stocks)} invalid")
    return valid_stocks, invalid_stocks

if __name__ == "__main__":
    try:
        # Create comprehensive Indian stock database
        new_stocks = create_stock_database()
        
        # Validate data availability
        valid_stocks, invalid_stocks = validate_stock_data()
        
        logger.info("=" * 60)
        logger.info("🎯 STOCK DATABASE SETUP COMPLETE")
        logger.info(f"📊 Total stocks: {Stock.objects.count()}")
        logger.info(f"✅ New stocks added: {new_stocks}")
        logger.info(f"🔍 Validation results: {len(valid_stocks)} working, {len(invalid_stocks)} issues")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"💥 Critical error in stock database setup: {str(e)}")
        raise
