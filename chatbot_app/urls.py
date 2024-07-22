from django.urls import path
from .views import get_asgi_application

urlpatterns = [
    path('', get_asgi_application()),  # Route all paths to FastAPI app
]
