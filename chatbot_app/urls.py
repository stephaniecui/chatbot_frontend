from django.urls import path
from .views import index, chatbot_response

urlpatterns = [
    path('', index, name='index'),  # Simple Django view for root
    path('response/', chatbot_response, name='chatbot_response'),
]
