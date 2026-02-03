"""
Quantum Fusion Layer for Q-FuseVision AI Lab

Uses PennyLane to create a 4-qubit Variational Quantum Circuit (VQC)
that fuses image and text embeddings into a quantum-enhanced representation.
"""

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. Using classical fusion fallback.")


class QuantumFusionLayer:
    """
    Quantum Fusion Layer using PennyLane.
    
    Creates a 4-qubit variational quantum circuit with:
    - RX and RY rotation gates for encoding
    - CNOT gates for entanglement
    - Measurement to produce fused embedding vector
    """
    
    def __init__(self, num_qubits=4, num_layers=2):
        """
        Initialize the Quantum Fusion Layer.
        
        Args:
            num_qubits: Number of qubits in the circuit (default: 4)
            num_layers: Number of variational layers (default: 2)
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        if PENNYLANE_AVAILABLE:
            # Create quantum device
            self.dev = qml.device('default.qubit', wires=num_qubits)
            
            # Create the quantum circuit as a QNode
            self.circuit = qml.QNode(self._quantum_circuit, self.dev)
        else:
            self.dev = None
            self.circuit = None
    
    def _quantum_circuit(self, inputs, weights):
        """
        Define the variational quantum circuit.
        
        Args:
            inputs: Input features to encode (normalized)
            weights: Trainable weights for variational gates
        
        Returns:
            Expectation values for each qubit (Pauli-Z measurements)
        """
        # Encode input features using angle encoding
        for i in range(self.num_qubits):
            qml.RX(inputs[i % len(inputs)] * np.pi, wires=i)
            qml.RY(inputs[(i + 1) % len(inputs)] * np.pi, wires=i)
        
        # Variational layers
        for layer in range(self.num_layers):
            # Rotation gates with trainable parameters
            for i in range(self.num_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            
            # Entangling CNOT gates (ring topology)
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])
            
            # Additional entanglement for depth
            if layer < self.num_layers - 1:
                for i in range(0, self.num_qubits - 1, 2):
                    qml.CZ(wires=[i, i + 1])
        
        # Measure all qubits in Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
    
    def _normalize_embedding(self, embedding, target_size):
        """
        Normalize and resize embedding to target size.
        
        Args:
            embedding: Input embedding array
            target_size: Desired output size
        
        Returns:
            Normalized embedding of target size
        """
        embedding = np.array(embedding).flatten()
        
        # Resize to target size
        if len(embedding) > target_size:
            # Downsample by averaging chunks
            chunk_size = len(embedding) // target_size
            resized = np.array([
                np.mean(embedding[i*chunk_size:(i+1)*chunk_size]) 
                for i in range(target_size)
            ])
        elif len(embedding) < target_size:
            # Upsample by interpolation
            indices = np.linspace(0, len(embedding) - 1, target_size)
            resized = np.interp(indices, np.arange(len(embedding)), embedding)
        else:
            resized = embedding
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(resized)) > 0:
            resized = resized / np.max(np.abs(resized))
        
        return resized
    
    def fuse_embeddings(self, image_embedding, text_embedding):
        """
        Fuse image and text embeddings using quantum circuit.
        
        Args:
            image_embedding: Embedding from image (numpy array or list)
            text_embedding: Embedding from text (numpy array or list)
        
        Returns:
            Fused embedding vector (list of floats)
        """
        # Convert to numpy arrays
        image_emb = np.array(image_embedding).flatten()
        text_emb = np.array(text_embedding).flatten()
        
        # Normalize embeddings to fit quantum circuit input
        image_normalized = self._normalize_embedding(image_emb, self.num_qubits)
        text_normalized = self._normalize_embedding(text_emb, self.num_qubits)
        
        # Combine embeddings
        combined = (image_normalized + text_normalized) / 2
        
        if PENNYLANE_AVAILABLE and self.circuit is not None:
            # Initialize random weights for variational circuit
            np.random.seed(42)  # For reproducibility
            weights = np.random.uniform(
                low=-np.pi, 
                high=np.pi, 
                size=(self.num_layers, self.num_qubits, 3)
            )
            
            # Run quantum circuit
            try:
                quantum_output = self.circuit(combined, weights)
                
                # Convert to regular Python floats
                fused_vector = [float(x) for x in quantum_output]
                
                # Expand the fused vector with additional classical processing
                expanded_vector = self._expand_quantum_output(
                    fused_vector, 
                    combined
                )
                
                return expanded_vector
                
            except Exception as e:
                print(f"Quantum circuit error: {e}")
                return self._classical_fallback(image_normalized, text_normalized)
        else:
            return self._classical_fallback(image_normalized, text_normalized)
    
    def _expand_quantum_output(self, quantum_output, classical_input):
        """
        Expand quantum output with classical post-processing.
        
        Args:
            quantum_output: Output from quantum circuit
            classical_input: Original combined input
        
        Returns:
            Expanded embedding vector
        """
        # Combine quantum and classical features
        expanded = []
        
        # Add quantum measurements
        expanded.extend(quantum_output)
        
        # Add classical statistics
        expanded.append(float(np.mean(classical_input)))
        expanded.append(float(np.std(classical_input)))
        expanded.append(float(np.max(classical_input)))
        expanded.append(float(np.min(classical_input)))
        
        # Add quantum-classical products
        for i, q_val in enumerate(quantum_output):
            expanded.append(float(q_val * classical_input[i]))
        
        # Normalize final vector
        expanded = np.array(expanded)
        if np.linalg.norm(expanded) > 0:
            expanded = expanded / np.linalg.norm(expanded)
        
        return [float(x) for x in expanded]
    
    def _classical_fallback(self, image_emb, text_emb):
        """
        Classical fallback when PennyLane is not available.
        
        Args:
            image_emb: Normalized image embedding
            text_emb: Normalized text embedding
        
        Returns:
            Classically fused embedding vector
        """
        # Simple classical fusion
        combined = np.concatenate([image_emb, text_emb])
        
        # Apply non-linear transformation
        transformed = np.tanh(combined)
        
        # Add statistical features
        stats = [
            float(np.mean(combined)),
            float(np.std(combined)),
            float(np.mean(image_emb)),
            float(np.mean(text_emb)),
        ]
        
        result = np.concatenate([transformed, stats])
        
        # Normalize
        if np.linalg.norm(result) > 0:
            result = result / np.linalg.norm(result)
        
        return [float(x) for x in result]


def create_image_embedding(image_path):
    """
    Create a simple embedding from an image.
    
    Uses basic image statistics as a simple embedding.
    In production, you would use a proper vision model.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Numpy array embedding
    """
    try:
        from PIL import Image
        
        # Load and process image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((64, 64))
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Create embedding from image statistics
        embedding = []
        
        # Channel-wise statistics
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            embedding.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
            ])
        
        # Spatial features (quadrant means)
        h, w = img_array.shape[:2]
        for i in range(2):
            for j in range(2):
                quadrant = img_array[
                    i*h//2:(i+1)*h//2, 
                    j*w//2:(j+1)*w//2
                ]
                embedding.append(np.mean(quadrant))
        
        return np.array(embedding)
        
    except Exception as e:
        print(f"Error creating image embedding: {e}")
        return np.random.randn(16)


def create_text_embedding(text):
    """
    Create a simple embedding from text.
    
    Uses basic text statistics as a simple embedding.
    In production, you would use a proper text model.
    
    Args:
        text: Input text string
    
    Returns:
        Numpy array embedding
    """
    try:
        # Basic text features
        words = text.lower().split()
        chars = list(text.lower())
        
        embedding = []
        
        # Length features
        embedding.append(len(text) / 1000.0)
        embedding.append(len(words) / 100.0)
        embedding.append(np.mean([len(w) for w in words]) / 10.0 if words else 0)
        
        # Character frequency features
        common_chars = 'etaoinshrdlcumwfgypbvkjxqz'
        for char in common_chars[:10]:
            embedding.append(chars.count(char) / max(len(chars), 1))
        
        # Question features
        embedding.append(1.0 if '?' in text else 0.0)
        embedding.append(1.0 if any(w in text.lower() for w in ['what', 'how', 'why', 'where', 'when', 'who']) else 0.0)
        embedding.append(1.0 if any(w in text.lower() for w in ['describe', 'explain', 'tell']) else 0.0)
        
        # Pad or truncate to fixed size
        target_size = 16
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        return np.array(embedding)
        
    except Exception as e:
        print(f"Error creating text embedding: {e}")
        return np.random.randn(16)


# Global quantum fusion instance
quantum_fusion = QuantumFusionLayer(num_qubits=4, num_layers=2)


def fuse_vision_language(image_path, question):
    """
    Main function to fuse vision and language inputs.
    
    Args:
        image_path: Path to the image file
        question: User's question about the image
    
    Returns:
        Fused embedding vector as a list of floats
    """
    # Create embeddings
    image_embedding = create_image_embedding(image_path)
    text_embedding = create_text_embedding(question)
    
    # Fuse using quantum circuit
    fused_vector = quantum_fusion.fuse_embeddings(image_embedding, text_embedding)
    
    return fused_vector
