"""
Admin configuration for Q-FuseVision AI Lab
"""

from django.contrib import admin
from .models import FusionExperiment


@admin.register(FusionExperiment)
class FusionExperimentAdmin(admin.ModelAdmin):
    """Admin configuration for FusionExperiment model."""
    
    list_display = [
        'id', 
        'question_preview', 
        'status', 
        'created_at', 
        'updated_at'
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['question', 'ollama_answer']
    readonly_fields = ['created_at', 'updated_at', 'fused_vector']
    
    fieldsets = (
        ('Input', {
            'fields': ('image', 'question')
        }),
        ('Processing', {
            'fields': ('status', 'error_message')
        }),
        ('Results', {
            'fields': ('vision_analysis', 'fused_vector', 'ollama_answer'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def question_preview(self, obj):
        """Return truncated question for list display."""
        return obj.question[:50] + '...' if len(obj.question) > 50 else obj.question
    question_preview.short_description = 'Question'
