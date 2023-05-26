from django.urls import path
from .views import PerfPrediction

urlpatterns = [
    path('perf/', PerfPrediction.as_view(http_method_names=['post']), name='perf_prediction'),
]