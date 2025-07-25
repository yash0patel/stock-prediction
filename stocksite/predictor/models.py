from django.db import models

class Stock(models.Model):
    ticker = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.ticker

class News(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    headline = models.CharField(max_length=256)
    published_date = models.DateTimeField()
    sentiment = models.FloatField()

    def __str__(self):
        return self.headline

class Prediction(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    date = models.DateField()
    result = models.CharField(max_length=10)  # "Up", "Down", "Neutral"

    def __str__(self):
        return f"{self.stock.ticker} {self.date}: {self.result}"
