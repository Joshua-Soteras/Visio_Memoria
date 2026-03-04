# Deployment & UI Strategy — Key Points

## Recommended Approach (Phased)

### Phase 1 — Now (Streamlit)
- Core pipeline (FAISS, SQLite, labeling UI) is not built yet
- Streamlit lets you iterate fast and validate logic
- Don't spend time on frontend infrastructure before the backend works

### Phase 2 — FastAPI
- Wrap the Python pipeline in a FastAPI backend
- Creates a clean separation between ML logic and UI
- Enables any frontend to talk to the pipeline over HTTP / WebSocket

### Phase 3 — React + Electron
- Once the pipeline is solid, migrate UI to React + Electron
- Gives full control over UI/UX — genuinely polished interface
- Electron packages the app as a native desktop binary (.dmg, .exe, Linux)

### Phase 4 — Deploy to Jetson Orin Nano
- Dedicated edge AI device, always on, fully offline
- CUDA acceleration replaces MPS (code already has CUDA fallback)
- Strong portfolio piece

---

## Why This Order
- Avoid being blocked on frontend work while ML pipeline is incomplete
- Validate the logic cheaply with Streamlit first
- Graduate to polished UI once the core is proven

---

## Deployment Architecture (Final State)

```
Electron (shell)
    └── React (frontend UI)
            ↕ HTTP / WebSocket (localhost)
        FastAPI (Python backend)
            → YOLOv8 → DINOv3/InsightFace → FAISS → SQLite
```

- Runs fully offline — no WiFi required after initial setup
- Webcam handled by Python/OpenCV, frames streamed to React via WebSocket
- Python bundled with PyInstaller, app packaged with electron-builder

---

## Portfolio Value
- Jetson + React + Electron = self-contained edge AI desktop app
- Fully offline facial recognition running on dedicated hardware
- Stands out compared to a standard laptop demo
