"""
Advanced Vision Processor for Q-FuseVision AI Lab

Implements:
- Object skeleton detection and marking
- Quantum-enhanced distance calculation
- Anomaly detection using isolation forest
- Edge detection and contour analysis
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import json
import os
import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    cv2 = importlib.import_module("cv2")
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False
    logger.info("OpenCV not available. Using PIL-based fallback.")

try:
    IsolationForest = importlib.import_module("sklearn.ensemble").IsolationForest
    StandardScaler = importlib.import_module("sklearn.preprocessing").StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not available. Anomaly detection disabled.")


class QuantumDistanceCalculator:
    """
    Quantum-enhanced distance calculation for object relationships.
    Uses quantum-inspired algorithms for improved spatial analysis.
    """
    
    def __init__(self):
        self.num_qubits = 4
        
    def quantum_distance_matrix(self, points):
        """
        Calculate quantum-enhanced distance matrix between points.
        Uses superposition-inspired averaging for robust distances.
        """
        n = len(points)
        if n == 0:
            return np.array([])
        
        points = np.array(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Euclidean base distance
                euclidean = np.linalg.norm(points[i] - points[j])
                
                # Quantum-inspired phase encoding
                phase = np.cos(np.pi * euclidean / (np.max(points) + 1e-6))
                
                # Amplitude encoding for scale invariance
                amplitude = np.exp(-euclidean / (np.mean(np.abs(points)) + 1e-6))
                
                # Quantum superposition-style combination
                quantum_dist = euclidean * (1 + 0.1 * phase) * (1 + 0.1 * amplitude)
                
                distances[i, j] = quantum_dist
                distances[j, i] = quantum_dist
        
        return distances
    
    def estimate_depth(self, object_features):
        """
        Estimate relative depth/distance of objects using quantum-inspired calculation.
        """
        depths = []
        for obj in object_features:
            # Use size and position to estimate depth
            size_factor = obj.get('area', 1) / obj.get('image_area', 1)
            y_position = obj.get('center_y', 0.5)
            
            # Larger objects and lower positions are typically closer
            # Quantum-inspired non-linear combination
            depth_score = np.exp(-size_factor * 5) * (1 - y_position * 0.5)
            depth_score = np.clip(depth_score, 0, 1)
            
            depths.append({
                'object_id': obj.get('id', 0),
                'relative_depth': float(depth_score),
                'confidence': float(1 - abs(0.5 - depth_score))
            })
        
        return depths


class AnomalyDetector:
    """
    Anomaly detection for identifying unusual regions in images.
    Uses Isolation Forest with quantum-enhanced feature extraction.
    """
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
        else:
            self.model = None
            self.scaler = None
    
    def extract_patch_features(self, image, patch_size=32):
        """
        Extract features from image patches for anomaly detection.
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        features = []
        positions = []
        
        for y in range(0, h - patch_size, patch_size // 2):
            for x in range(0, w - patch_size, patch_size // 2):
                patch = img_array[y:y+patch_size, x:x+patch_size]
                
                # Extract patch features
                patch_features = []
                
                if len(patch.shape) == 3:
                    for c in range(min(3, patch.shape[2])):
                        channel = patch[:, :, c]
                        patch_features.extend([
                            np.mean(channel),
                            np.std(channel),
                            np.median(channel),
                            np.percentile(channel, 10),
                            np.percentile(channel, 90),
                        ])
                else:
                    patch_features.extend([
                        np.mean(patch),
                        np.std(patch),
                        np.median(patch),
                        np.percentile(patch, 10),
                        np.percentile(patch, 90),
                    ])
                
                # Texture features (gradient-based)
                dy = np.gradient(patch.mean(axis=2) if len(patch.shape) == 3 else patch, axis=0)
                dx = np.gradient(patch.mean(axis=2) if len(patch.shape) == 3 else patch, axis=1)
                patch_features.extend([
                    np.mean(np.abs(dx)),
                    np.mean(np.abs(dy)),
                    np.std(dx),
                    np.std(dy),
                ])
                
                features.append(patch_features)
                positions.append((x + patch_size // 2, y + patch_size // 2))
        
        return np.array(features), positions
    
    def detect_anomalies(self, image):
        """
        Detect anomalous regions in the image.
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            return {
                'anomaly_regions': [],
                'anomaly_score': 0.0,
                'message': 'Anomaly detection unavailable (scikit-learn not installed)'
            }
        
        features, positions = self.extract_patch_features(image)
        
        if len(features) < 5:
            return {
                'anomaly_regions': [],
                'anomaly_score': 0.0,
                'message': 'Image too small for anomaly detection'
            }
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit and predict
        self.model.fit(features_scaled)
        predictions = self.model.predict(features_scaled)
        scores = self.model.decision_function(features_scaled)
        
        # Find anomalous regions
        anomaly_indices = np.where(predictions == -1)[0]
        anomaly_regions = []
        
        for idx in anomaly_indices:
            x, y = positions[idx]
            anomaly_regions.append({
                'x': int(x),
                'y': int(y),
                'score': float(-scores[idx]),  # Higher score = more anomalous
            })
        
        # Overall anomaly score
        overall_score = float(np.mean(-scores[anomaly_indices])) if len(anomaly_indices) > 0 else 0.0
        
        return {
            'anomaly_regions': anomaly_regions,
            'anomaly_score': overall_score,
            'num_anomalies': len(anomaly_regions),
            'total_patches': len(features)
        }


class ObjectSkeletonDetector:
    """
    Detect object skeletons and mark main features.
    Uses edge detection and morphological operations.
    """
    
    def __init__(self):
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
    
    def detect_edges_pil(self, image):
        """
        Detect edges using PIL (fallback when OpenCV not available).
        """
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Enhance edges
        enhancer = ImageEnhance.Contrast(edges)
        edges = enhancer.enhance(2.0)
        
        return np.array(edges)
    
    def detect_edges_cv2(self, image):
        """
        Detect edges using OpenCV Canny detector.
        """
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
        
        return edges
    
    def detect_contours(self, image):
        """
        Detect object contours/outlines.
        """
        if CV2_AVAILABLE:
            edges = self.detect_edges_cv2(image)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            edges = self.detect_edges_pil(image)
            # Simple contour detection for PIL
            contours = self._find_contours_simple(edges)
        
        return contours, edges
    
    def _find_contours_simple(self, edge_array):
        """
        Simple contour finding for PIL-based edge detection.
        """
        # Threshold the edge image
        binary = (edge_array > 50).astype(np.uint8)
        
        # Find connected components (simplified contours)
        contours = []
        visited = np.zeros_like(binary)
        
        for y in range(binary.shape[0]):
            for x in range(binary.shape[1]):
                if binary[y, x] and not visited[y, x]:
                    contour = self._trace_contour(binary, visited, x, y)
                    if len(contour) > 10:  # Minimum contour size
                        contours.append(np.array(contour))
        
        return contours
    
    def _trace_contour(self, binary, visited, start_x, start_y):
        """
        Trace a contour starting from a point.
        """
        contour = []
        stack = [(start_x, start_y)]
        
        while stack and len(contour) < 1000:
            x, y = stack.pop()
            
            if (0 <= x < binary.shape[1] and 
                0 <= y < binary.shape[0] and 
                binary[y, x] and 
                not visited[y, x]):
                
                visited[y, x] = 1
                contour.append([[x, y]])
                
                # Add neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((x + dx, y + dy))
        
        return contour
    
    def extract_skeleton_features(self, image):
        """
        Extract skeleton features from detected contours.
        """
        contours, edges = self.detect_contours(image)
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        image_area = h * w
        
        objects = []
        
        for i, contour in enumerate(contours):
            if len(contour) < 5:
                continue
            
            # Get contour properties
            contour_array = np.array(contour)
            
            if CV2_AVAILABLE:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Bounding box
                x, y, bw, bh = cv2.boundingRect(contour)
                
                # Centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + bw // 2, y + bh // 2
            else:
                # Simplified calculations for PIL
                points = contour_array.reshape(-1, 2)
                x, y = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                bw, bh = x2 - x, y2 - y
                area = bw * bh * 0.7  # Approximate
                perimeter = 2 * (bw + bh)
                cx, cy = x + bw // 2, y + bh // 2
            
            # Skip very small or very large objects
            if area < 100 or area > image_area * 0.9:
                continue
            
            # Calculate shape features
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            aspect_ratio = bw / (bh + 1e-6)
            extent = area / (bw * bh + 1e-6)
            
            objects.append({
                'id': i,
                'area': float(area),
                'image_area': float(image_area),
                'perimeter': float(perimeter),
                'bounding_box': {'x': int(x), 'y': int(y), 'width': int(bw), 'height': int(bh)},
                'center_x': float(cx / w),
                'center_y': float(cy / h),
                'center_px': (int(cx), int(cy)),
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
                'extent': float(extent),
                'contour_points': len(contour),
            })
        
        return objects, edges


class AdvancedVisionProcessor:
    """
    Main vision processor combining all analysis components.
    """
    
    def __init__(self):
        self.skeleton_detector = ObjectSkeletonDetector()
        self.distance_calculator = QuantumDistanceCalculator()
        self.anomaly_detector = AnomalyDetector()
    
    def process_image(self, image_path):
        """
        Perform comprehensive image analysis.
        """
        try:
            # Load image
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            # Resize for consistent processing
            max_size = 800
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            results = {
                'image_info': {
                    'width': image.size[0],
                    'height': image.size[1],
                    'mode': image.mode,
                    'path': str(image_path),
                },
                'processing_status': 'success',
            }
            
            # 1. Skeleton/Object Detection
            objects, edge_map = self.skeleton_detector.extract_skeleton_features(image)
            results['detected_objects'] = objects
            results['num_objects'] = len(objects)
            
            # 2. Quantum Distance Calculation
            if objects:
                centers = [(obj['center_px'][0], obj['center_px'][1]) for obj in objects]
                distance_matrix = self.distance_calculator.quantum_distance_matrix(centers)
                results['distance_matrix'] = distance_matrix.tolist() if len(distance_matrix) > 0 else []
                
                # Depth estimation
                depths = self.distance_calculator.estimate_depth(objects)
                results['depth_estimation'] = depths
            else:
                results['distance_matrix'] = []
                results['depth_estimation'] = []
            
            # 3. Anomaly Detection
            anomaly_results = self.anomaly_detector.detect_anomalies(image)
            results['anomaly_analysis'] = anomaly_results
            
            # 4. Generate summary for LLM
            results['analysis_summary'] = self._generate_analysis_summary(results)
            
            return results
            
        except Exception as e:
            return {
                'processing_status': 'error',
                'error_message': str(e),
            }
    
    def _generate_analysis_summary(self, results):
        """
        Generate a human-readable summary of the analysis.
        """
        summary_parts = []
        
        # Image info
        img_info = results.get('image_info', {})
        summary_parts.append(
            f"Image dimensions: {img_info.get('width', 'unknown')}x{img_info.get('height', 'unknown')} pixels"
        )
        
        # Object detection summary
        num_objects = results.get('num_objects', 0)
        summary_parts.append(f"Detected {num_objects} distinct object region(s)")
        
        if num_objects > 0:
            objects = results.get('detected_objects', [])
            
            # Categorize objects by size
            large = [o for o in objects if o['area'] / o['image_area'] > 0.1]
            medium = [o for o in objects if 0.01 < o['area'] / o['image_area'] <= 0.1]
            small = [o for o in objects if o['area'] / o['image_area'] <= 0.01]
            
            if large:
                summary_parts.append(f"  - {len(large)} large object(s) (main subjects)")
            if medium:
                summary_parts.append(f"  - {len(medium)} medium-sized object(s)")
            if small:
                summary_parts.append(f"  - {len(small)} small object(s)/details")
            
            # Shape analysis
            circular = [o for o in objects if o['circularity'] > 0.7]
            elongated = [o for o in objects if o['aspect_ratio'] > 2 or o['aspect_ratio'] < 0.5]
            
            if circular:
                summary_parts.append(f"  - {len(circular)} circular/rounded shape(s)")
            if elongated:
                summary_parts.append(f"  - {len(elongated)} elongated/rectangular shape(s)")
        
        # Depth analysis
        depths = results.get('depth_estimation', [])
        if depths:
            foreground = [d for d in depths if d['relative_depth'] < 0.3]
            background = [d for d in depths if d['relative_depth'] > 0.7]
            
            if foreground:
                summary_parts.append(f"Objects in foreground: {len(foreground)}")
            if background:
                summary_parts.append(f"Objects in background: {len(background)}")
        
        # Anomaly analysis
        anomaly = results.get('anomaly_analysis', {})
        num_anomalies = anomaly.get('num_anomalies', 0)
        if num_anomalies > 0:
            summary_parts.append(
                f"Detected {num_anomalies} unusual region(s) "
                f"(anomaly score: {anomaly.get('anomaly_score', 0):.2f})"
            )
        
        return "\n".join(summary_parts)
    
    def create_annotated_image(self, image_path, output_path=None):
        """
        Create an annotated version of the image showing detected objects.
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Resize for consistent processing
            max_size = 800
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            draw = ImageDraw.Draw(image)
            
            # Get detected objects
            objects, _ = self.skeleton_detector.extract_skeleton_features(image)
            
            # Draw bounding boxes and centers
            colors = ['#00FF00', '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#F38181']
            
            for i, obj in enumerate(objects):
                color = colors[i % len(colors)]
                bbox = obj['bounding_box']
                
                # Draw bounding box
                draw.rectangle(
                    [(bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])],
                    outline=color,
                    width=2
                )
                
                # Draw center point
                cx, cy = obj['center_px']
                draw.ellipse(
                    [(cx - 5, cy - 5), (cx + 5, cy + 5)],
                    fill=color
                )
                
                # Draw object ID
                draw.text((bbox['x'], bbox['y'] - 15), f"Obj {i+1}", fill=color)
            
            # Save or return
            if output_path:
                image.save(output_path)
                return output_path
            else:
                return image
                
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return None


# Global instance
vision_processor = AdvancedVisionProcessor()


def analyze_image_advanced(image_path):
    """
    Main function to perform advanced image analysis.
    """
    return vision_processor.process_image(image_path)


def create_annotated_image(image_path, output_path=None):
    """
    Create an annotated version of the image showing detected objects with bounding boxes.
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save the annotated image
        
    Returns:
        Path to annotated image or PIL Image object
    """
    return vision_processor.create_annotated_image(image_path, output_path)


def get_detection_overlay_data(image_path):
    """
    Get detection data for JavaScript canvas overlay rendering.
    
    Returns a dictionary containing:
    - detected_objects: List of objects with bounding boxes
    - image_dimensions: Original image dimensions
    - anomaly_regions: Detected anomaly areas
    - depth_data: Depth estimation for each object
    """
    try:
        analysis = vision_processor.process_image(image_path)
        
        return {
            'success': True,
            'detected_objects': analysis.get('detected_objects', []),
            'num_objects': analysis.get('num_objects', 0),
            'image_info': analysis.get('image_info', {}),
            'anomaly_analysis': analysis.get('anomaly_analysis', {}),
            'depth_estimation': analysis.get('depth_estimation', []),
            'distance_matrix': analysis.get('distance_matrix', []),
            'analysis_summary': analysis.get('analysis_summary', ''),
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'detected_objects': [],
            'num_objects': 0,
        }

