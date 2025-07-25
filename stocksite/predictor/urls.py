from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # path('api/predict/<str:ticker>/', views.PredictView.as_view()),
    path('api/predict/', views.PredictView.as_view(), name='predict'),
]
