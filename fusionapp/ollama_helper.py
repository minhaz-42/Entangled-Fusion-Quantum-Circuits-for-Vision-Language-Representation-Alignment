"""
Ollama Helper for Q-FuseVision AI Lab

Safely calls Ollama models (llava for vision, mistral for reasoning)
using subprocess and HTTP API for local model inference.
"""

import subprocess
import json
import base64
import os
import shutil
import urllib.request
import urllib.error
from pathlib import Path


class OllamaHelper:
    """
    Helper class for interacting with Ollama local models.
    
    Supports:
    - LLaVA for vision-language analysis (via HTTP API)
    - Mistral for text reasoning
    """
    
    def __init__(self, host='http://localhost:11434'):
        """
        Initialize the Ollama helper.
        
        Args:
            host: Ollama API host (default: http://localhost:11434)
        """
        self.host = host
        self.vision_model = 'llava'
        self.text_model = 'mistral'
        self.timeout = 180  # seconds - increased for vision processing
    
    def is_ollama_available(self):
        """Check if Ollama is installed and running."""
        try:
            # Check if ollama command exists
            if not shutil.which('ollama'):
                return False, "Ollama is not installed. Please install from https://ollama.ai"
            
            # Check if ollama API is responding
            try:
                req = urllib.request.Request(f'{self.host}/api/tags', method='GET')
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        return True, "Ollama is available and running"
            except urllib.error.URLError:
                # Try to start ollama or check via CLI
                result = subprocess.run(
                    ['ollama', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return True, "Ollama is available"
                else:
                    return False, "Ollama is installed but not running. Please run 'ollama serve'"
            
            return True, "Ollama is available"
                
        except subprocess.TimeoutExpired:
            return False, "Ollama connection timed out"
        except FileNotFoundError:
            return False, "Ollama command not found"
        except Exception as e:
            return False, f"Error checking Ollama: {str(e)}"
    
    def list_models(self):
        """List available Ollama models."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                models = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                return models
            return []
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def has_model(self, model_name):
        """Check if a specific model is available."""
        models = self.list_models()
        return any(model_name in m for m in models)
    
    def pull_model(self, model_name):
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
        
        Returns:
            Tuple of (success, message)
        """
        try:
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for large models
            )
            
            if result.returncode == 0:
                return True, f"Successfully pulled {model_name}"
            else:
                return False, f"Failed to pull {model_name}: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, f"Timeout pulling {model_name}"
        except Exception as e:
            return False, f"Error pulling {model_name}: {str(e)}"
    
    def _encode_image_base64(self, image_path):
        """
        Encode an image to base64 string.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Base64 encoded string
        """
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def analyze_image(self, image_path, question):
        """
        Analyze an image using LLaVA model via HTTP API.
        
        Args:
            image_path: Path to the image file
            question: Question about the image
        
        Returns:
            Tuple of (success, response_text or error_message)
        """
        try:
            # Verify image exists
            if not os.path.exists(image_path):
                return False, f"Image not found: {image_path}"
            
            # First try HTTP API (most reliable for vision)
            success, result = self._analyze_image_http_api(image_path, question)
            if success:
                return True, result
            
            # Fallback to CLI with image flag
            print(f"HTTP API failed ({result}), trying CLI...")
            return self._analyze_image_cli(image_path, question)
                
        except Exception as e:
            return False, f"Vision analysis error: {str(e)}"
    
    def _analyze_image_http_api(self, image_path, question):
        """
        Analyze image using Ollama HTTP API (most reliable method).
        """
        try:
            # Encode image to base64
            image_base64 = self._encode_image_base64(image_path)
            
            # Build prompt for detailed analysis
            prompt = f"""Look at this image carefully and analyze it in detail.

{question}

Provide a specific, detailed response based ONLY on what you can actually see in this exact image. 
Do not make assumptions or describe things that aren't visible.
Focus on: the main subject(s), their appearance, colors, positions, and any notable details."""
            
            # Create API request payload
            request_data = {
                "model": self.vision_model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more accurate descriptions
                    "num_predict": 1024,
                }
            }
            
            # Make HTTP request
            data = json.dumps(request_data).encode('utf-8')
            req = urllib.request.Request(
                f'{self.host}/api/generate',
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_text = response.read().decode('utf-8')
                
                # Parse response - might be single JSON or newline-delimited
                try:
                    response_data = json.loads(response_text)
                    if 'response' in response_data:
                        return True, response_data['response']
                    elif 'error' in response_data:
                        return False, response_data['error']
                except json.JSONDecodeError:
                    # Handle streaming format (newline-delimited JSON)
                    full_response = ''
                    for line in response_text.strip().split('\n'):
                        try:
                            chunk = json.loads(line)
                            full_response += chunk.get('response', '')
                        except:
                            pass
                    
                    if full_response:
                        return True, full_response
                    return False, "Could not parse vision response"
            
        except urllib.error.URLError as e:
            return False, f"Connection error: {str(e)}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP error: {e.code} - {e.reason}"
        except Exception as e:
            return False, f"API error: {str(e)}"
    
    def _analyze_image_cli(self, image_path, question):
        """
        Fallback: Analyze image using Ollama CLI.
        """
        try:
            prompt = f"""Analyze this image and answer: {question}
            
Describe exactly what you see in the image. Be specific and detailed about:
- The main subject(s)
- Colors, shapes, and positions
- Any text or notable features
- The overall scene"""
            
            # Use multimodal format with --image flag
            result = subprocess.run(
                ['ollama', 'run', self.vision_model, prompt],
                input=image_path,  # Some versions accept this
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, 'OLLAMA_IMAGE': image_path}
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return True, result.stdout.strip()
            else:
                error = result.stderr.strip() if result.stderr else "Empty response"
                return False, f"CLI error: {error}"
                
        except Exception as e:
            return False, f"CLI analysis error: {str(e)}"
    
    def reason_with_context(self, vision_analysis, question, fused_vector=None, advanced_analysis=None):
        """
        Use Mistral to reason about the vision analysis.
        
        Args:
            vision_analysis: Text from vision model analysis
            question: Original user question
            fused_vector: Optional quantum-fused embedding for context
            advanced_analysis: Optional dict from advanced vision processor
        
        Returns:
            Tuple of (success, response_text or error_message)
        """
        try:
            # Build comprehensive prompt
            context_parts = []
            
            # Quantum fusion context
            if fused_vector:
                vector_stats = {
                    'dimensions': len(fused_vector),
                    'mean': sum(fused_vector) / len(fused_vector) if fused_vector else 0,
                    'max': max(fused_vector) if fused_vector else 0,
                    'min': min(fused_vector) if fused_vector else 0,
                }
                context_parts.append(f"""
Quantum Fusion Analysis:
- Embedding dimensions: {vector_stats['dimensions']}
- Mean activation: {vector_stats['mean']:.4f}
- Activation range: [{vector_stats['min']:.4f}, {vector_stats['max']:.4f}]""")
            
            # Advanced vision analysis context
            if advanced_analysis and advanced_analysis.get('processing_status') == 'success':
                summary = advanced_analysis.get('analysis_summary', '')
                if summary:
                    context_parts.append(f"""
Computer Vision Analysis:
{summary}""")
                
                # Object detection details
                objects = advanced_analysis.get('detected_objects', [])
                if objects:
                    context_parts.append(f"""
Object Detection Results:
- Total objects detected: {len(objects)}""")
                    for i, obj in enumerate(objects[:5]):  # Top 5 objects
                        size_pct = (obj['area'] / obj['image_area']) * 100
                        context_parts.append(
                            f"  - Object {i+1}: {size_pct:.1f}% of image, "
                            f"aspect ratio {obj['aspect_ratio']:.2f}, "
                            f"circularity {obj['circularity']:.2f}"
                        )
                
                # Anomaly detection
                anomaly = advanced_analysis.get('anomaly_analysis', {})
                if anomaly.get('num_anomalies', 0) > 0:
                    context_parts.append(f"""
Anomaly Detection:
- {anomaly['num_anomalies']} unusual regions detected
- Anomaly score: {anomaly.get('anomaly_score', 0):.3f}""")
            
            context_text = "\n".join(context_parts)
            
            prompt = f"""You are an expert AI assistant analyzing vision-language fusion results.

ORIGINAL QUESTION: {question}

VISION MODEL ANALYSIS:
{vision_analysis}
{context_text}

Based on the above analysis, provide a comprehensive and accurate answer to the original question.
Focus on what was actually observed in the image analysis.
Be specific, detailed, and highlight key observations. Format your response clearly with sections if needed."""
            
            # Run mistral for reasoning
            result = subprocess.run(
                ['ollama', 'run', self.text_model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if response:
                    return True, response
                else:
                    return False, "Empty response from Mistral"
            else:
                return False, f"Reasoning error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Reasoning timed out. The model is taking too long."
        except Exception as e:
            return False, f"Reasoning error: {str(e)}"
    
    def simple_generate(self, prompt, model=None):
        """
        Simple text generation with specified model.
        
        Args:
            prompt: Text prompt
            model: Model to use (default: mistral)
        
        Returns:
            Tuple of (success, response_text or error_message)
        """
        try:
            model = model or self.text_model
            
            result = subprocess.run(
                ['ollama', 'run', model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, f"Generation error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Generation timed out"
        except Exception as e:
            return False, f"Generation error: {str(e)}"


# Global Ollama helper instance
ollama_helper = OllamaHelper()


def process_experiment(image_path, question, fused_vector=None, advanced_analysis=None):
    """
    Process a complete experiment: vision analysis + reasoning.
    
    Args:
        image_path: Path to the image file
        question: User's question
        fused_vector: Optional quantum-fused embedding
        advanced_analysis: Optional advanced vision processor results
    
    Returns:
        Dictionary with results or error information
    """
    results = {
        'success': False,
        'vision_analysis': None,
        'final_answer': None,
        'error': None
    }
    
    # Check Ollama availability
    available, message = ollama_helper.is_ollama_available()
    if not available:
        results['error'] = message
        return results
    
    # Check if LLaVA model is available
    if not ollama_helper.has_model('llava'):
        results['error'] = "LLaVA model not found. Please run: ollama pull llava"
        return results
    
    # Step 1: Vision analysis with LLaVA
    success, vision_result = ollama_helper.analyze_image(image_path, question)
    
    if not success:
        # Log the error but continue with advanced analysis
        print(f"Vision model warning: {vision_result}")
        results['vision_analysis'] = f"[Vision model had issues: {vision_result}]"
        
        # Use advanced analysis summary as fallback
        if advanced_analysis and advanced_analysis.get('analysis_summary'):
            vision_result = advanced_analysis['analysis_summary']
        else:
            vision_result = f"User asked about an image: {question}"
    else:
        results['vision_analysis'] = vision_result
    
    # Step 2: Check if Mistral is available
    if not ollama_helper.has_model('mistral'):
        # Just return vision analysis if no reasoning model
        results['final_answer'] = vision_result
        results['success'] = True
        return results
    
    # Step 3: Reasoning with Mistral
    success, reasoning_result = ollama_helper.reason_with_context(
        vision_result or f"User question about an image: {question}",
        question,
        fused_vector,
        advanced_analysis
    )
    
    if success:
        results['final_answer'] = reasoning_result
        results['success'] = True
    else:
        results['error'] = reasoning_result
        # Provide the vision analysis as fallback
        if results['vision_analysis']:
            results['final_answer'] = results['vision_analysis']
            results['success'] = True
    
    return results
