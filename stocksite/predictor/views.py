import sys
import os
import logging
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings

# Ensure project root on path
project_root = settings.BASE_DIR
if project_root not in sys.path:
    sys.path.append(project_root)

# Import pipeline functions
from predictor.prepare_features import process_stock
from predictor.train_model import train_one
from predictor.models import Stock

logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'predictor/home.html')

def normalize_ticker(ticker):
    t = ticker.strip().upper()
    if '.' not in t:
        t += '.NS'
    return t

class PredictView(APIView):
    def get(self, request):
        raw = request.GET.get('ticker')
        if not raw:
            return render(request, 'predictor/home.html')

        ticker = normalize_ticker(raw)
        feature_file = os.path.join(settings.BASE_DIR, 'data', f"{ticker}_features.csv")
        model_file = os.path.join(settings.BASE_DIR, 'models', f"{ticker}_best_model.pkl")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(feature_file), exist_ok=True)
        os.makedirs(os.path.dirname(model_file), exist_ok=True)

        # ⭐ KEY FIX: Auto-generate missing files instead of 404 error
        stock_obj, _ = Stock.objects.get_or_create(ticker=ticker, defaults={'name': ticker})
        
        if not os.path.exists(feature_file) or not os.path.exists(model_file):
            logger.info(f"🔄 Generating data and model for {ticker}")
            try:
                # Step 1: Generate features (download data, create technical indicators)
                success = process_stock(stock_obj)
                if not success:
                    return render(request, 'predictor/home.html', {
                        'error': f'❌ Failed to fetch data for {ticker}. Please check if it\'s a valid Indian stock symbol.'
                    })
                
                # Step 2: Train ML model
                if os.path.exists(feature_file):
                    train_one(feature_file)
                else:
                    return render(request, 'predictor/home.html', {
                        'error': f'❌ Feature generation failed for {ticker}'
                    })
                    
            except Exception as e:
                logger.error(f"Error generating data for {ticker}: {e}")
                return render(request, 'predictor/home.html', {
                    'error': f'❌ Error processing {ticker}: {str(e)}'
                })

        # Validate files exist after generation
        if not os.path.exists(feature_file):
            return render(request, 'predictor/home.html', {
                'error': f'❌ Features for {ticker} could not be generated.'
            })
        if not os.path.exists(model_file):
            return render(request, 'predictor/home.html', {
                'error': f'❌ Model for {ticker} could not be trained.'
            })

        try:
            # Load feature data
            df = pd.read_csv(feature_file, parse_dates=['Date'])
            latest = df.iloc[-1:]
            X = latest.drop(columns=[c for c in ['Date', 'Movement', 'NextClose', 'ReturnNext'] if c in df.columns])

            # Load trained model components
            obj = joblib.load(model_file)
            model, scaler, le = obj['model'], obj['scaler'], obj['le']

            # Make prediction
            X_scaled = scaler.transform(X)
            probas = model.predict_proba(X_scaled)
            idx = probas.argmax()
            label = le.inverse_transform([idx])
            confidence = float(probas[idx])

            # Calculate difficulty level
            difficulty = 'low'
            if confidence < 0.6:
                difficulty = 'medium'
            if confidence < 0.4:
                difficulty = 'high'

            # Generate trend chart
            hist = df[['Date', 'Movement']].tail(20).copy()
            hist['Value'] = hist['Movement'].map({'Down': -1, 'Neutral': 0, 'Up': 1})
            plt.figure(figsize=(10, 4))
            plt.plot(hist['Date'], hist['Value'], marker='o')
            plt.yticks([-1, 0, 1], ['Down', 'Neutral', 'Up'])
            plt.title(f"{ticker} Last 20 Movements")
            plt.xticks(rotation=45)
            
            # Save chart
            fname = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            path = os.path.join(settings.MEDIA_ROOT, fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            chart_url = settings.MEDIA_URL + fname

            result = {
                "ticker": ticker,
                "date": latest['Date'].dt.strftime('%Y-%m-%d').values,
                "predicted": label,
                "confidence": round(confidence, 3),
                "difficulty": difficulty,
                "chart_url": chart_url
            }

            # Return JSON or HTML response
            if request.GET.get('format') == 'json':
                return Response(result)
            return render(request, 'predictor/home.html', result)
            
        except Exception as e:
            logger.error(f"Prediction error for {ticker}: {e}")
            return render(request, 'predictor/home.html', {
                'error': f'❌ Prediction failed for {ticker}: {str(e)}'
            })
