
from django.urls import path
from .views import index, chatbot_response

urlpatterns = [
    path('', index, name='index'),
    path('chat/', chatbot_response, name='chatbot_response'),
]
