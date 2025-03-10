from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static 

app_name = 'ivr_apis'

urlpatterns = [
    path("", views.IVR_APIS.as_view(), name="test"),
]

# Serve static and media files in development mode
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
