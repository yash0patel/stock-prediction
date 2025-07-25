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
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings

from django.http import HttpResponse

def home(request):
    return HttpResponse("Indian Stock Predictor: App setup complete.")


# Set up logging
logger = logging.getLogger(__name__)

def normalize_ticker(ticker):
    t = ticker.strip().upper()
    if not t.endswith('.NS'):
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

        if not os.path.exists(feature_file):
            return JsonResponse({"error": f"Features for {ticker} not found."}, status=404)
        if not os.path.exists(model_file):
            return JsonResponse({"error": f"Model for {ticker} not found."}, status=404)

        # Load features
        df = pd.read_csv(feature_file, parse_dates=['Date'])
        latest = df.iloc[-1:]
        X = latest.drop(columns=[c for c in ['Date','Movement','NextClose','ReturnNext'] if c in df.columns])

        # Load model, scaler, label encoder
        obj = joblib.load(model_file)
        model, scaler, le = obj['model'], obj['scaler'], obj['le']

        # Scale and predict
        X_scaled = scaler.transform(X)
        probas = model.predict_proba(X_scaled)[0]
        idx = probas.argmax()
        label = le.inverse_transform([idx])[0]
        confidence = float(probas[idx])

        # Difficulty heuristic
        difficulty = 'low'
        if confidence < 0.6:
            difficulty = 'medium'
        if confidence < 0.4:
            difficulty = 'high'

        # Last 20 movements chart
        hist = df[['Date','Movement']].tail(20).copy()
        hist['Value'] = hist['Movement'].map({'Down':-1,'Neutral':0,'Up':1})
        plt.figure(figsize=(10,4))
        plt.plot(hist['Date'], hist['Value'], marker='o')
        plt.yticks([-1,0,1], ['Down','Neutral','Up'])
        plt.title(f"{ticker} Last 20 Movements")
        plt.xticks(rotation=45)
        fname = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        path = os.path.join(settings.MEDIA_ROOT, fname)
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        chart_url = settings.MEDIA_URL + fname

        result = {
            "ticker": ticker,
            "date": latest['Date'].dt.strftime('%Y-%m-%d').values[0],
            "predicted": label,
            "confidence": round(confidence, 3),
            "difficulty": difficulty
        }

        if request.GET.get('format') == 'json' or request.headers.get('Accept') == 'application/json':
            return Response(result)

        return render(request, 'predictor/home.html', {
            **result,
            'chart_url': chart_url
        })
