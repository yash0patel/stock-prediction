
from django.urls import path
from . import views

urlpatterns = [
    # Make PredictView handle the root URL '' directly
    path('', views.PredictView.as_view(), name='predict'),
    path('predict/', views.PredictView.as_view(), name='predict'),
]
