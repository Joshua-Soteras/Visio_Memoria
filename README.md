# Visio_Memoria

Facial recognition system that detects, remembers, and greets people it has seen before. 





--- 

### System Pipeline 

Portable Camera (laptop/phone/Jetson)
    │
    ▼
┌─────────────────────┐
│ Face Detection       │  ← YOLOv8-face or RetinaFace
│ + Person Detection   │    (also detect full body for activity)
└──────┬──────────────┘
       │
       ├── cropped + aligned face
       │         │
       │         ▼
       │   ┌─────────────────┐
       │   │ DINOv3 ViT-B/16 │  → face embedding (768-d)
       │   │ (frozen)        │
       │   └──────┬──────────┘
       │          │
       │          ▼
       │   ┌─────────────────┐
       │   │ FAISS Index     │  cosine similarity search
       │   │ (~1000 people)  │
       │   └──────┬──────────┘
       │          │
       │          ├── MATCH → "Welcome back, Alex!"
       │          │           log visit timestamp
       │          │
       │          └── NO MATCH → save crop + embedding
       │                        queue for labeling
       │
       ├── scene crop (full body + surroundings)
       │         │
       │         ▼
       │   ┌─────────────────┐
       │   │ DINOv3 ViT-B/16 │  → scene embedding
       │   │ (same model!)   │    (patch features for activity)
       │   └──────┬──────────┘
       │          │
       │          ▼
       │   ┌─────────────────────────┐
       │   │ Activity Classifier     │  lightweight linear head
       │   │ or LLM captioning       │  on DINOv3 features
       │   └─────────────────────────┘
       │
       ▼
┌──────────────────────┐
│ SQLite Database       │
│ - persons (id, name, embedding, face_crop)
│ - visits (person_id, timestamp, activity, location)
│ - unknowns (embedding, crop, cluster_id)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Labeling UI          │  Streamlit app, end of day
│ "Who is this?"       │  shows clustered unknown faces
└──────────────────────┘ 


--- 

### Active python env 
source .venv/bin/activate

--- 

## Learning YOLO 

- assuming model = YOLO(load_model)
- What is returned from model() is a Results object
- model can return a list of results 

- .boxes	-> Boxes Object	-> (Most Important) Contains bounding box coordinates, confidence scores, and class IDs. Used in Object Detection & Instance Segmentation.
- .keypoints	Keypoints Object	Contains x, y coordinates (and visibility) for specific landmarks. Crucial for your face model (eyes, nose, mouth).
- .masks	Masks Object	Contains segmentation masks (pixel-level outlines). None for your face model unless it is a segmentation model.
- .probs	Probs Object	Contains classification probabilities (e.g., "99% chance this whole image is a cat"). Used only in Classification tasks.
- .obb	OBB Object	Oriented Bounding Boxes (rotated boxes), used for aerial imagery or angled objects.

--- 
Learning CV
- pixel coordinates must be whole numbers 

---

### References
- https://github.com/derronqi/yolov8-face?tab=readme-ov-file 