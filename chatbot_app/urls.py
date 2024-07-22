from django.urls import path
from .views import fastapi_app

urlpatterns = [
    path('', fastapi_app),  # Route all paths to FastAPI app
]
