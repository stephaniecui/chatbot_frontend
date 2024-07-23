from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Main page
    path('/chatbot/chat/', views.chatbot_response, name='chatbot_response'),  # Chatbot response endpoint
]
