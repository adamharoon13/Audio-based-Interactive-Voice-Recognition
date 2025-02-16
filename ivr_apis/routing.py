from django.urls import path
from .consumers import IVRConsumer

websocket_urlpatterns = [
    path("ws/ivr/", IVRConsumer.as_asgi()),
]
