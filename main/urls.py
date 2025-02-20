from django.urls import path
from .views import *

urlpatterns = [
    path('',LoginView.as_view(),name='log'),
    path('register/',RegView.as_view(),name='reg'),
    path('home/',Home.as_view(),name='h'),
    path('prediction/',PredictionView.as_view(),name='predict'),
    path('doctors/',DoctorsView.as_view(),name='doc'),
    path('groq/',get_groq_response,name='groq'),
    path('logout/',custom_logout,name="logout"),
    path('chat-history/', HistoryView.as_view(), name='chat_history'),
    path('all-history/', HistoryAllView.as_view(), name='all_history'),
    path('download-report/<str:filename>/', download_pdf, name='download_report'),
    path('delete/<str:chat_id>/', delete_chat, name='del'),
]