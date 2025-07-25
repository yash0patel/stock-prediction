from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use('Agg')   # Prevent GUI issues
import matplotlib.pyplot as plt
from django.conf import settings
from datetime import datetime

def home(request):
    return HttpResponse("Indian Stock Predictor: App setup complete.")

class PredictView(APIView):
    def get(self, request):
        ticker = request.GET.get("ticker", None)
        if not ticker:
            return render(request, 'predictor/home.html')

        feature_path = f"{ticker}_features.csv"
        if not os.path.exists(feature_path):
            return JsonResponse({"error": f"Features file for {ticker} not found"}, status=404)

        model_path = f"models/{ticker}_model.pkl"
        if not os.path.exists(model_path):
            return JsonResponse({"error": f"Model for {ticker} not found"}, status=404)

        # Load data & model
        df = pd.read_csv(feature_path)
        df['Date'] = pd.to_datetime(df['Date'])  # FIX for date parsing

        row = df.iloc[-1:]
        model = joblib.load(model_path)

        # Prediction
        X = row.drop(['Date', 'Movement', 'Future_Close', 'Future_Return'], axis=1)
        pred = model.predict(X)[0]
        pred_mapping = {0: "Neutral", 1: "Up", 2: "Down"}
        pred_label = pred_mapping.get(pred, str(pred))
        date = row.Date.dt.strftime('%Y-%m-%d').values[0]

        # Handle Movement column mapping
        movement_data = df['Movement'].tail(20)
        if movement_data.dtype == object:
            y_values = movement_data.map({"Down": -1, "Neutral": 0, "Up": 1})
        else:
            # Assume already numeric (-1,0,1)
            y_values = movement_data

        # Chart
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'].tail(20), y_values, label="Actual", color="blue")
        plt.axhline(0, color='black', linestyle='--')
        plt.yticks([-1, 0, 1], ['Down', 'Neutral', 'Up'])
        plt.xticks(rotation=45)
        plt.title(f"{ticker} Last 20 Movements")
        plt.legend()

        chart_filename = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}_chart.png"
        chart_path = os.path.join(settings.MEDIA_ROOT, chart_filename)
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        chart_url = settings.MEDIA_URL + chart_filename

        if request.headers.get('Accept') == 'application/json' or request.GET.get('format') == 'json':
            return Response({"ticker": ticker, "date": date, "predicted": pred_label})

        return render(request, 'predictor/home.html', {
            'predicted': pred_label,
            'ticker': ticker,
            'date': date,
            'chart_url': chart_url
        })
