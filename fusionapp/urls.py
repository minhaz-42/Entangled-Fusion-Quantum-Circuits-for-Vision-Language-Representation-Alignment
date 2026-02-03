"""
URL configuration for fusionapp.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.landing, name='landing'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_experiment, name='upload'),
    path('results/<int:experiment_id>/', views.results, name='results'),
    
    # Authentication
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Actions
    path('delete/<int:experiment_id>/', views.delete_experiment, name='delete_experiment'),
    
    # API endpoints
    path('api/status/<int:experiment_id>/', views.api_experiment_status, name='api_experiment_status'),
    path('api/ollama/', views.api_ollama_status, name='api_ollama_status'),
    path('api/detection/<int:experiment_id>/', views.api_detection_data, name='api_detection_data'),
    path('api/reanalyze/<int:experiment_id>/', views.api_reanalyze, name='api_reanalyze'),
    path('annotated/<int:experiment_id>/', views.serve_annotated_image, name='annotated_image'),
]
