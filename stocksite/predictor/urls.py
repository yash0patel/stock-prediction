
from django.urls import path
from . import views

urlpatterns = [
    # Make PredictView handle the root URL '' directly
    path('', views.PredictView.as_view(), name='predict'),
    path('predict/', views.PredictView.as_view(), name='predict'),
    path('how-it-works/', views.how_it_works, name='how_it_works'),
    path('contact/', views.contact_us, name='contact_us'),
]
