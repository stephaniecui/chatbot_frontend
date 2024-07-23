"""
ASGI config for chatbot_project project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from chatbot_app.views import fastapi_app

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": URLRouter([
        path("chatbot/chat/", fastapi_app),
        path("", get_asgi_application()),
    ]),
})
