"""
Views for Q-FuseVision AI Lab

Handles all page rendering and experiment processing.
"""

import json
import traceback
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required

from .models import FusionExperiment
from .forms import FusionExperimentForm, CustomUserCreationForm
from .quantum_fusion import fuse_vision_language
from .ollama_helper import process_experiment, ollama_helper
from .vision_processor import analyze_image_advanced, vision_processor, create_annotated_image


def landing(request):
    """
    Landing page view.
    
    Displays the welcome page with introduction to Q-FuseVision AI Lab.
    """
    # Get some stats for the landing page
    total_experiments = FusionExperiment.objects.count()
    completed_experiments = FusionExperiment.objects.filter(status='completed').count()
    
    context = {
        'total_experiments': total_experiments,
        'completed_experiments': completed_experiments,
        'page_title': 'Welcome',
    }
    
    return render(request, 'landing.html', context)


def register_view(request):
    """User registration view."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.username}! Your account has been created.')
            return redirect('dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'auth/register.html', {'form': form, 'page_title': 'Register'})


def login_view(request):
    """User login view."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            next_url = request.GET.get('next', 'dashboard')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    
    return render(request, 'auth/login.html', {'form': form, 'page_title': 'Login'})


def logout_view(request):
    """User logout view."""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('landing')


def dashboard(request):
    """
    Dashboard view.
    
    Shows overview and list of past experiments.
    """
    # Get all experiments
    experiments_list = FusionExperiment.objects.all()
    
    # Pagination
    paginator = Paginator(experiments_list, 10)
    page_number = request.GET.get('page', 1)
    experiments = paginator.get_page(page_number)
    
    # Statistics
    stats = {
        'total': FusionExperiment.objects.count(),
        'completed': FusionExperiment.objects.filter(status='completed').count(),
        'pending': FusionExperiment.objects.filter(status='pending').count(),
        'processing': FusionExperiment.objects.filter(status='processing').count(),
        'failed': FusionExperiment.objects.filter(status='failed').count(),
    }
    
    # Check Ollama status
    ollama_available, ollama_message = ollama_helper.is_ollama_available()
    
    context = {
        'experiments': experiments,
        'stats': stats,
        'ollama_available': ollama_available,
        'ollama_message': ollama_message,
        'page_title': 'Dashboard',
    }
    
    return render(request, 'dashboard.html', context)


def upload_experiment(request):
    """
    Upload experiment view.
    
    Handles image upload and question submission, then processes
    through quantum fusion and Ollama models.
    """
    if request.method == 'POST':
        form = FusionExperimentForm(request.POST, request.FILES)
        
        if form.is_valid():
            # Save the experiment
            experiment = form.save(commit=False)
            if request.user.is_authenticated:
                experiment.user = request.user
            experiment.status = 'processing'
            experiment.save()
            
            try:
                # Get the image path
                image_path = experiment.image.path
                question = experiment.question
                
                # Step 1: Advanced Vision Processing (skeleton, anomaly, object detection)
                try:
                    advanced_analysis = analyze_image_advanced(image_path)
                    # Save analysis summary
                    if advanced_analysis.get('processing_status') == 'success':
                        experiment.advanced_analysis = json.dumps(advanced_analysis)
                except Exception as e:
                    print(f"Advanced vision warning: {e}")
                    advanced_analysis = None
                
                # Step 2: Quantum Fusion
                try:
                    fused_vector = fuse_vision_language(image_path, question)
                    experiment.set_fused_vector_list(fused_vector)
                except Exception as e:
                    print(f"Quantum fusion warning: {e}")
                    fused_vector = [0.0] * 8  # Fallback vector
                    experiment.set_fused_vector_list(fused_vector)
                
                # Step 3: Ollama Processing with advanced analysis
                results = process_experiment(
                    image_path, 
                    question, 
                    fused_vector,
                    advanced_analysis
                )
                
                if results['success']:
                    experiment.vision_analysis = results.get('vision_analysis', '')
                    experiment.ollama_answer = results.get('final_answer', '')
                    experiment.status = 'completed'
                else:
                    experiment.error_message = results.get('error', 'Unknown error')
                    # Still save partial results if available
                    if results.get('vision_analysis'):
                        experiment.vision_analysis = results['vision_analysis']
                    if results.get('final_answer'):
                        experiment.ollama_answer = results['final_answer']
                        experiment.status = 'completed'
                    else:
                        experiment.status = 'failed'
                
                experiment.save()
                
                messages.success(
                    request, 
                    'Experiment processed successfully!'
                )
                return redirect('results', experiment_id=experiment.pk)
                
            except Exception as e:
                experiment.status = 'failed'
                experiment.error_message = str(e)
                experiment.save()
                
                messages.error(
                    request, 
                    f'Error processing experiment: {str(e)}'
                )
                return redirect('results', experiment_id=experiment.pk)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = FusionExperimentForm()
    
    # Check Ollama status
    ollama_available, ollama_message = ollama_helper.is_ollama_available()
    
    context = {
        'form': form,
        'ollama_available': ollama_available,
        'ollama_message': ollama_message,
        'page_title': 'New Experiment',
    }
    
    return render(request, 'upload.html', context)


def results(request, experiment_id):
    """
    Results view.
    
    Displays the results of a completed experiment including
    image preview, fused vector visualization, and Ollama answer.
    """
    experiment = get_object_or_404(FusionExperiment, pk=experiment_id)
    
    # Get vector data for visualization
    vector_data = experiment.get_vector_visualization_data()
    
    context = {
        'experiment': experiment,
        'vector_data': json.dumps(vector_data),
        'vector_list': experiment.get_fused_vector_list(),
        'page_title': f'Results #{experiment_id}',
    }
    
    return render(request, 'results.html', context)


@require_http_methods(["POST"])
def delete_experiment(request, experiment_id):
    """
    Delete an experiment.
    """
    experiment = get_object_or_404(FusionExperiment, pk=experiment_id)
    
    # Delete associated image file
    if experiment.image:
        try:
            experiment.image.delete(save=False)
        except Exception:
            pass
    
    experiment.delete()
    
    messages.success(request, 'Experiment deleted successfully.')
    return redirect('dashboard')


@require_http_methods(["GET"])
def api_experiment_status(request, experiment_id):
    """
    API endpoint to check experiment status.
    """
    experiment = get_object_or_404(FusionExperiment, pk=experiment_id)
    
    return JsonResponse({
        'id': experiment.pk,
        'status': experiment.status,
        'has_answer': bool(experiment.ollama_answer),
        'error': experiment.error_message,
    })


@require_http_methods(["GET"])
def api_ollama_status(request):
    """
    API endpoint to check Ollama availability.
    """
    available, message = ollama_helper.is_ollama_available()
    models = ollama_helper.list_models() if available else []
    
    return JsonResponse({
        'available': available,
        'message': message,
        'models': models,
    })


@require_http_methods(["GET"])
def api_detection_data(request, experiment_id):
    """
    API endpoint to get object detection data for an experiment.
    Returns detection data for JavaScript canvas overlay rendering.
    """
    experiment = get_object_or_404(FusionExperiment, pk=experiment_id)
    
    # Try to get from stored analysis first
    advanced_analysis = experiment.get_advanced_analysis()
    
    if advanced_analysis and advanced_analysis.get('detected_objects'):
        return JsonResponse({
            'success': True,
            'experiment_id': experiment.pk,
            'detected_objects': advanced_analysis.get('detected_objects', []),
            'num_objects': advanced_analysis.get('num_objects', 0),
            'image_info': advanced_analysis.get('image_info', {}),
            'anomaly_analysis': advanced_analysis.get('anomaly_analysis', {}),
            'depth_estimation': advanced_analysis.get('depth_estimation', []),
            'analysis_summary': advanced_analysis.get('analysis_summary', ''),
        })
    
    # If no stored analysis, return empty
    return JsonResponse({
        'success': False,
        'experiment_id': experiment.pk,
        'message': 'No detection data available',
        'detected_objects': [],
        'num_objects': 0,
    })


@require_http_methods(["GET"])
def api_reanalyze(request, experiment_id):
    """
    API endpoint to re-run object detection on an experiment's image.
    """
    experiment = get_object_or_404(FusionExperiment, pk=experiment_id)
    
    if not experiment.image:
        return JsonResponse({
            'success': False,
            'error': 'No image found for this experiment',
        })
    
    try:
        from .vision_processor import get_detection_overlay_data
        
        result = get_detection_overlay_data(experiment.image.path)
        
        # Update stored analysis
        if result.get('success'):
            experiment.advanced_analysis = json.dumps(result)
            experiment.save()
        
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
        })


@require_http_methods(["GET"])
def serve_annotated_image(request, experiment_id):
    """
    Serve an annotated version of the experiment image with bounding boxes.
    """
    from django.http import HttpResponse
    import io
    
    experiment = get_object_or_404(FusionExperiment, pk=experiment_id)
    
    if not experiment.image:
        return HttpResponse('No image found', status=404)
    
    try:
        # Create annotated image
        annotated = create_annotated_image(experiment.image.path)
        
        if annotated is None:
            return HttpResponse('Failed to create annotated image', status=500)
        
        # Convert to bytes
        buffer = io.BytesIO()
        annotated.save(buffer, format='PNG')
        buffer.seek(0)
        
        return HttpResponse(buffer.getvalue(), content_type='image/png')
        
    except Exception as e:
        return HttpResponse(f'Error: {str(e)}', status=500)

