# Q-FuseVision — Entangled Fusion Quantum Circuits for Vision–Language Alignment 🔬⚡

Q-FuseVision is an experimental Django application that combines advanced computer vision processing, a quantum-inspired fusion layer, and local LLM reasoning to answer natural-language questions about uploaded images. It includes a modern UI (landing, dashboard, upload, auth), a detailed vision pipeline, and optional integrations for PennyLane (quantum VQC) and Ollama (LLaVA + Mistral) for multimodal analysis and reasoning.

---

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Quick Start (development)](#quick-start-development)
- [Optional Integrations](#optional-integrations)
  - [PennyLane (quantum fusion)](#pennylane-quantum-fusion)
  - [Ollama (local LLMs)](#ollama-local-llms)
- [Usage & Endpoints](#usage--endpoints)
- [Development notes & troubleshooting](#development-notes--troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- Interactive results visualizer with canvas-based bounding boxes, center markers, and overlays.
- Advanced vision processing: skeleton/contour detection, object detection, depth estimation, anomaly detection (IsolationForest), and summarization.
- Quantum-inspired fusion module that combines visual and language embeddings (PennyLane optional; falls back to a classical fusion if not installed).
- Multimodal reasoning pipeline using Ollama (LLaVA for image analysis and Mistral for reasoning) when available locally.
- Simple REST endpoints for programmatic use (`/api/detection/`, `/api/reanalyze/`, `/annotated/`) and a user-facing dashboard.

---

## Tech Stack
- Python 3.10+ (tested with 3.11)
- Django 5.2
- Pillow / OpenCV (for image processing)
- Optional: PennyLane (quantum fusion), Ollama (local LLM server)

Files to know:
- `fusionapp/quantum_fusion.py` — Quantum fusion logic & classical fallback
- `fusionapp/vision_processor.py` — Advanced vision pipeline
- `fusionapp/ollama_helper.py` — Ollama integration and model orchestration

---

## Requirements
- Clone the repo
- Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start (development)
1. Apply database migrations:

```bash
python3 manage.py migrate
```

2. Create an admin user:

```bash
python3 manage.py createsuperuser
```

3. Run the development server:

```bash
python3 manage.py runserver
```

Open http://127.0.0.1:8000/ to access the app.

---

## Optional Integrations

### PennyLane (quantum fusion)
- PennyLane is optional. If not installed, the app uses a classical fusion fallback and prints:

```
Warning: PennyLane not available. Using classical fusion fallback.
```

- To enable the quantum VQC path, install PennyLane and a fast device (optional):

```bash
pip install "pennylane>=0.33.0" "pennylane-lightning>=0.33.0"
```

- The fusion code is in `fusionapp/quantum_fusion.py` and will use PennyLane if available.

### Ollama (local LLMs)
- Ollama provides local multimodal models. The helper expects Ollama to be installed and an API available at `http://localhost:11434`.
- Install Ollama and run the daemon (see https://ollama.ai):

```bash
# Example (follow Ollama docs for platform-specific steps)
ollama serve
ollama pull llava
ollama pull mistral
```

- `fusionapp/ollama_helper.py` checks for `ollama` and makes calls to the local API. If models are missing, the helper will ask you to run `ollama pull llava` or `ollama pull mistral`.

---

## Usage & Endpoints
- Upload image + question via the web UI (Upload page) to create a `FusionExperiment`.
- Key endpoints:
  - `/api/detection/` — detection data
  - `/api/reanalyze/` — re-run analysis for an experiment
  - `/annotated/` — serve annotated images

- Workflow:
  1. Upload image and question
  2. Server runs `analyze_image_advanced()` → `fuse_vision_language()` → `process_experiment()`
  3. Ollama (LLaVA) performs vision-language analysis; Mistral performs reasoning if available

---

## Development notes & troubleshooting
- If you see: `Warning: PennyLane not available. Using classical fusion fallback.` — install PennyLane as shown above to enable quantum fusion.
- If Ollama is not installed or not running, the dashboard will report Ollama as offline. Run `ollama serve` and `ollama pull llava`/`mistral` to enable full multimodal behavior.
- Database: If you add new migrations, run `python3 manage.py migrate`.

---

## Contributing
- Please open issues or pull requests. Prefer small, focused changes and include tests where appropriate.
- Code style: follow Black/flake8 if you add CI.

---

## License
MIT — see the `LICENSE` file.

---

If you'd like, I can also:
- Add a short demo GIF or screenshots to `README.md` ✅
- Add contributing templates / issue templates ✅
- Add a `dev` Makefile or `docker-compose` for easy local setup ✅

If you want any of those, tell me which and I'll add it next.
