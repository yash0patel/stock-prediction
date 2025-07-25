from django.contrib import admin
from .models import Stock, News, Prediction

admin.site.register(Stock)
admin.site.register(News)
admin.site.register(Prediction)