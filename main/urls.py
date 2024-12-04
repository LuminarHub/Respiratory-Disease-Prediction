from django.urls import path
from .views import *

urlpatterns = [
    path('',LoginView.as_view(),name='log'),
    path('register/',RegView.as_view(),name='reg'),
    path('home/',Home.as_view(),name='h'),
    path('prediction/',PredictionView.as_view(),name='predict'),
    path('logout/',custom_logout,name="logout")
]