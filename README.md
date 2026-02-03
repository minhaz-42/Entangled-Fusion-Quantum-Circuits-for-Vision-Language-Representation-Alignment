# Entangled-Fusion-Quantum-Circuits-for-Vision-Language-Representation-Alignment

Q-FuseVision — A cyberpunk-styled Django lab that fuses computer vision and language with quantum-inspired techniques to deliver interactive, explainable image analysis and vision-language reasoning.

## Summary

Q-FuseVision is an experimental web platform where users upload images and natural-language queries; the system runs advanced vision analysis (object detection, depth estimation, anomaly detection), performs a quantum-inspired fusion of visual + language features, and reasons with LLM-based models (Ollama/LLaVA + Mistral) to produce rich, interactive results.

## Key Features

- Interactive results page with canvas-drawn bounding boxes, center markers, labels and edge overlays.
- Advanced vision pipeline: skeleton/contour detection, anomaly detection (IsolationForest), quantum-inspired distance & depth estimation.
- Quantum fusion module (PennyLane fallback supported) to combine vision and language embeddings.
- Ollama integration for multimodal reasoning (vision prompt + fused context).
- APIs for detection data, re-analysis, and serving annotated images (`/api/detection/`, `/api/reanalyze/`, `/annotated/`).
- Modern cyberpunk UI: landing, dashboard, upload, auth pages with a cohesive design system.

## Tech Stack

- Django 5.2.7
- Python, PIL / OpenCV (optional)
- PennyLane (optional)
- Ollama (LLaVA + Mistral)

## Quick Start (dev)

```bash
pip install -r requirements.txt
python3 manage.py migrate
python3 manage.py createsuperuser
python3 manage.py runserver
```

## License

MIT
