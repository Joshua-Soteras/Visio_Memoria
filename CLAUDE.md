# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Visio Memoria is a facial recognition and activity tracking system. The pipeline: YOLOv8-Face detects faces/landmarks → DINOv3 ViT-B/16 extracts 768-d embeddings → (planned) FAISS similarity search + SQLite persistence + Streamlit labeling UI.

Currently in prototype/integration phase — model loading, webcam inference, and benchmarking are working. Database, FAISS index, activity classifier, and labeling UI are not yet implemented.

## Commands

```bash
# Environment uses UV with Python 3.14
source .venv/bin/activate

# Run main entry point (currently a template)
python main.py

# Run webcam + YOLOv8 face detection demo (press 'q' to quit)
python src/visio_memoria/utils/webcam_test.py

# Run DINOv3 benchmark (latency, memory, parameter counts)
python src/visio_memoria/utils/dinov3_test.py

# Run hardware detection
python src/visio_memoria/utils/system_specs_reader.py
```

No test framework (pytest/unittest) is configured. Test scripts in `utils/` are standalone benchmarks run directly.

## Architecture

```
src/visio_memoria/
├── models/
│   ├── yolov8-face/          # Forked from derronqi/yolov8-face
│   │   ├── yolov8n-face.pt   # Pre-trained weights (~339MB)
│   │   └── ultralytics/      # Modified YOLOv8 implementation
│   └── dinov3/               # Meta DINOv3 research fork
│       ├── hubconf.py        # torch.hub.load() entry point
│       └── *.pth             # Pre-trained weights (~342MB)
└── utils/
    ├── dinov3_test.py        # DINOv3 benchmarking (latency, memory, model size)
    ├── webcam_test.py        # Real-time webcam + YOLO demo
    ├── system_specs_reader.py # Hardware/GPU detection
    └── temp.py               # YOLO landmarks learning notes
```

Both model directories are forked repos with their own internal structure. They are loaded locally (not from PyPI).

## Key Patterns

- **Model loading**: DINOv3 loads via `torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local')`. YOLOv8 loads via `YOLO("path/to/yolov8n-face.pt")`.
- **Path resolution**: Always uses `os.path.dirname(os.path.abspath(__file__))` for relative model/asset paths.
- **Device selection**: MPS (Apple Silicon) → CUDA → CPU fallback pattern. MPS/CUDA require explicit `synchronize()` calls for accurate timing.
- **Image preprocessing**: torchvision `v2.Compose` pipelines with ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). DINOv3 expects 256×256 inputs (multiple of 16 patch size).
- **Inference**: Always use `torch.inference_mode()` (faster than `torch.no_grad()`). Warmup runs before benchmarking to avoid cold-start artifacts.
- **Code style**: Heavy inline comments explaining PyTorch/ML concepts — this is intentional for learning purposes. Preserve this style.

## Configuration

- `.env` contains model download URLs
- `.python-version` specifies Python 3.14
- Model weights (`.pt`, `.pth`) are gitignored — not committed to the repo
