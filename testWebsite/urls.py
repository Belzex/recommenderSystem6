from django.urls import path
from .views import home, recommendation

urlpatterns = [path('', home),
               path('recommendation', recommendation)]