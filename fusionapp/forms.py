"""
Forms for Q-FuseVision AI Lab

Handles image upload and question input for fusion experiments.
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import FusionExperiment


class CustomUserCreationForm(UserCreationForm):
    """
    Custom user registration form with email field.
    """
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email',
        })
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Choose a username',
            }),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Create a password',
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm your password',
        })
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


class FusionExperimentForm(forms.ModelForm):
    """
    Form for creating a new fusion experiment.
    
    Allows users to upload an image and ask a question about it.
    """
    
    class Meta:
        model = FusionExperiment
        fields = ['image', 'question']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control file-input',
                'accept': 'image/*',
                'id': 'image-upload',
            }),
            'question': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Ask a question about the image... e.g., "What objects are in this image?" or "Describe the scene in detail."',
                'id': 'question-input',
            }),
        }
        labels = {
            'image': 'Upload Image',
            'question': 'Your Question',
        }
        help_texts = {
            'image': 'Supported formats: JPG, PNG, GIF, WebP (max 10MB)',
            'question': 'Ask any question about the uploaded image',
        }
    
    def clean_image(self):
        """Validate the uploaded image."""
        image = self.cleaned_data.get('image')
        
        if image:
            # Check file size (max 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError(
                    'Image file too large. Maximum size is 10MB.'
                )
            
            # Check file extension
            valid_extensions = ['jpg', 'jpeg', 'png', 'gif', 'webp']
            ext = image.name.split('.')[-1].lower()
            if ext not in valid_extensions:
                raise forms.ValidationError(
                    f'Unsupported file format. Use: {", ".join(valid_extensions)}'
                )
        
        return image
    
    def clean_question(self):
        """Validate the question."""
        question = self.cleaned_data.get('question', '').strip()
        
        if len(question) < 5:
            raise forms.ValidationError(
                'Question too short. Please provide more detail.'
            )
        
        if len(question) > 1000:
            raise forms.ValidationError(
                'Question too long. Maximum 1000 characters.'
            )
        
        return question
