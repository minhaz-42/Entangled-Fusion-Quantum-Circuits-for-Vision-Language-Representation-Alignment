"""
Models for Q-FuseVision AI Lab

FusionExperiment: Stores experiment data including image, question,
quantum-fused embedding vector, and Ollama AI response.
"""

import json
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User


class FusionExperiment(models.Model):
    """
    Model representing a Vision-Language fusion experiment.
    
    Stores the uploaded image, user question, quantum-fused embedding vector,
    and the AI-generated answer from Ollama models.
    """
    
    # User who created the experiment (optional for anonymous users)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='experiments'
    )
    
    # Image upload field
    image = models.ImageField(
        upload_to='uploads/%Y/%m/%d/',
        help_text='Upload an image for vision-language analysis'
    )
    
    # User's question about the image
    question = models.TextField(
        max_length=1000,
        help_text='Enter your question about the image'
    )
    
    # Quantum-fused embedding vector (stored as JSON)
    fused_vector = models.TextField(
        blank=True,
        null=True,
        help_text='Quantum-fused embedding vector (JSON format)'
    )
    
    # Advanced vision analysis (stored as JSON)
    advanced_analysis = models.TextField(
        blank=True,
        null=True,
        help_text='Advanced vision analysis results (JSON format)'
    )
    
    # Vision analysis from llava model
    vision_analysis = models.TextField(
        blank=True,
        null=True,
        help_text='Vision analysis from LLaVA model'
    )
    
    # Final answer from Ollama (mistral reasoning)
    ollama_answer = models.TextField(
        blank=True,
        null=True,
        help_text='Final reasoning answer from Mistral model'
    )
    
    # Processing status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    
    # Error message if processing failed
    error_message = models.TextField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Fusion Experiment'
        verbose_name_plural = 'Fusion Experiments'
    
    def __str__(self):
        return f"Experiment #{self.pk} - {self.question[:50]}..."
    
    def get_fused_vector_list(self):
        """Return the fused vector as a Python list."""
        if self.fused_vector:
            try:
                return json.loads(self.fused_vector)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_fused_vector_list(self, vector_list):
        """Set the fused vector from a Python list."""
        self.fused_vector = json.dumps(vector_list)
    
    def get_vector_visualization_data(self):
        """Return vector data formatted for visualization."""
        vector = self.get_fused_vector_list()
        return [
            {'index': i, 'value': float(v)} 
            for i, v in enumerate(vector)
        ]
    
    def get_advanced_analysis(self):
        """Return the advanced analysis as a Python dict."""
        if self.advanced_analysis:
            try:
                return json.loads(self.advanced_analysis)
            except json.JSONDecodeError:
                return {}
        return {}
