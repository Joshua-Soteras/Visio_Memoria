# Face Embedder — Concepts & Lessons Learned

Notes on the concepts behind `FaceEmbedder.py`, written as a learning reference.
Each topic maps to something implemented in the code.

---

## The Data Flow (Big Picture)

```
OpenCV Frame (BGR numpy array)
    │
    ▼
YOLOv8-face → bbox [x1, y1, x2, y2]
    │
    ▼
crop_face_from_frame() → PIL Image (RGB)
    │
    ▼
TRANSFORM pipeline → 4D Tensor [1, 3, 256, 256]
    │
    ▼
DINOv3 forward pass → raw embedding [1, 768]
    │
    ▼
F.normalize() → L2-normalized 1D vector [768]
    │
    ▼
.cpu() → stored in system RAM / passed to FAISS / DB
```

Understanding this flow makes every step make sense.
Each transformation exists because the *next* step demands a specific format.

---

## YOLO — What the Model Returns

When you call `model(frame)`, it returns a `Results` object (actually a list of them).

| Attribute    | Type             | What it holds                                         |
|--------------|------------------|-------------------------------------------------------|
| `.boxes`     | Boxes Object     | Bounding box coords, confidence scores, class IDs     |
| `.keypoints` | Keypoints Object | (x, y, visibility) for facial landmarks (eyes, nose)  |
| `.masks`     | Masks Object     | Pixel-level outlines — None unless segmentation model |
| `.probs`     | Probs Object     | Classification probabilities (not used here)           |
| `.obb`       | OBB Object       | Rotated bounding boxes for angled objects              |

**`results[0]`** is the first (usually only) result for a frame.
**`results[0].boxes`** stacks one row per detected face.
**`bbox = box.xyxy[0].tolist()`** extracts `[x1, y1, x2, y2]` as a Python list.

**bbox coordinate system:**
```
[x1, y1, x2, y2]
 ↑    ↑    ↑    ↑
 Left Top Right Bottom  (all pixel coordinates — must be integers)
```

---

## Cropping a Face — `crop_face_from_frame()`

The method converts a YOLO bbox into a PIL Image the embedder can consume.

**Key steps:**
1. Extract `height, width` from `frame.shape[:2]` (note: `frame.shape` is `[H, W, C]`)
2. Cast bbox coords to `int` — pixel indices must be whole numbers
3. Add padding (20% by default) to capture hair, ears, chin context
4. Clamp with `max(0, ...)` and `min(width/height, ...)` to prevent going out of frame bounds
5. Slice: `frame[y1:y2, x1:x2]` — NumPy slicing is `[rows, cols]` → `[y, x]`
6. Flip channels: `[:, :, ::-1]` converts BGR (OpenCV default) → RGB (what PIL/PyTorch expect)
7. `Image.fromarray(face_rgb)` — turns the raw NumPy array into a PIL Image object

**Why PIL?** Modern AI libraries (PyTorch, CLIP, etc.) expect PIL Images as input,
not raw NumPy arrays. `Image.fromarray()` is the standard bridge.

---

## The Transform Pipeline — `TRANSFORM`

`v2.Compose([...])` is a callable — it acts like a function that chains transforms in order.
Each step exists because the *next* one (or the model) requires a specific format.

```python
TRANSFORM = v2.Compose([
    v2.ToImage(),                                    # PIL/NumPy → Tensor, shape: (H,W,3) → (3,H,W)
    v2.Resize((256, 256), antialias=True),           # DINOv3 requires multiples of 16
    v2.ToDtype(torch.float32, scale=True),           # 0–255 integers → 0.0–1.0 floats
    v2.Normalize(mean=(0.485, 0.456, 0.406),         # Match ImageNet training stats
                 std=(0.229, 0.224, 0.225)),
])
```

**`v2.ToImage()`**
- Converts PIL or NumPy to a PyTorch Tensor
- Changes shape from `(H, W, 3)` → `(3, H, W)` — PyTorch is "channels first"

**`v2.Resize()`**
- DINOv3 uses a patch size of 16 — input dimensions must be multiples of 16
- `antialias=True` prevents jagged pixel artifacts when downscaling

**`v2.ToDtype(torch.float32, scale=True)`**
- `scale=True` divides by 255, converting `[0, 255]` integers to `[0.0, 1.0]` floats
- Neural nets work in floats — integer pixel values don't work well with gradients

**`v2.Normalize()`**
- Shifts and scales pixel values to match what DINOv3 saw during ImageNet training
- Formula per channel: `pixel = (pixel - mean) / std`
- **Getting this wrong produces garbage embeddings** — the model's internal weights assume this normalization

---

## Batching — Why `.unsqueeze(0)` Exists

Deep learning models are designed to process *batches* of images at once (for efficiency).
Because of this, they strictly expect a 4D input tensor, never 3D.

```
Single image:   [Channels, Height, Width]         → 3D — model rejects this
Batch of 1:     [Batch=1, Channels, Height, Width] → 4D — model accepts this
```

`.unsqueeze(0)` inserts a new dimension at position 0, wrapping your single image
into a "batch of 1" without copying any data.

In `get_embeddings_batch()`, `torch.stack([...])` does this for multiple images at once,
creating a true batch of shape `[N, 3, 256, 256]`.

---

## Model Loading — `torch.hub.load()`

`torch.hub.load()` is PyTorch's standard way to load models from a local repo or GitHub.
It looks for a `hubconf.py` file in the given directory, which defines what models are available.

```python
model = torch.hub.load(
    REPO_DIR,           # path to the dinov3 fork directory
    'dinov3_vitb16',    # the function name defined in hubconf.py
    source='local',     # don't try to download from GitHub
    weights=WEIGHTS     # path to the .pth weights file
)
```

After loading, always call `model.eval()` before inference (it's called inside `torch.hub.load`
for DINOv3, but worth knowing why).

---

## `eval()` vs `no_grad()` vs `inference_mode()`

These three are different — they do different things and are often used together.

| Call                      | What it does                                                              |
|---------------------------|---------------------------------------------------------------------------|
| `model.eval()`            | Switches layer *behavior* — disables Dropout, fixes BatchNorm stats       |
| `torch.no_grad()`         | Disables gradient tracking (saves memory). Now mostly replaced.           |
| `torch.inference_mode()`  | Disables gradients AND version tracking. Strictest, fastest for inference |

**In practice:** always call `model.eval()` once after loading, then wrap all
forward passes with `torch.inference_mode()`.

---

## Embeddings and L2 Normalization

A face embedding is a 768-dimensional vector — a point in 768D space.
DINOv3 outputs raw embeddings whose *direction* encodes identity but whose
*magnitude* (length) is arbitrary.

**L2 normalization** scales every vector to length 1 (onto the unit hypersphere).
After normalization:
- All embeddings have the same length — magnitude no longer affects comparisons
- **Cosine similarity = dot product** — the angle between two vectors is all that matters

```python
embedding = F.normalize(embedding[0], p=2, dim=0)
```

- `embedding[0]` — strips the batch dimension (shape `[1, 768]` → `[768]`)
- `p=2` — use L2 norm (Euclidean: square values, sum, take sqrt)
- `dim=0` — normalize across the only dimension of the 1D vector

For batch normalization in `get_embeddings_batch()`, `dim=1` is used instead
because the batch dimension is 0 and features are along dimension 1.

---

## `F.normalize` vs Batch/Layer Normalization

These sound similar but are completely different.

| Operation              | What it does                                                                     |
|------------------------|----------------------------------------------------------------------------------|
| `F.normalize(p=2)`     | Scales a *vector* to unit length. Post-processing step, not a trainable layer.  |
| Batch Normalization    | Normalizes features *across* a batch (mean/variance over N samples per feature)  |
| Layer Normalization    | Normalizes features *within* a single sample (across all features for one input) |

DINOv3 uses Layer Normalization internally. `F.normalize` is used *after* the model
as a post-processing step to make cosine similarity search work correctly.

---

## Why `.cpu()` at the End

`tensor.to(self.device)` moves data to GPU (MPS/CUDA) for fast computation.
`.cpu()` at the very end brings the result back to system RAM.

**Two reasons this matters:**

1. **VRAM is scarce** — GPU memory is small and expensive. Leaving embeddings on the GPU
   for thousands of faces will crash with an OOM error. Returning to CPU immediately
   frees VRAM for the next frame.

2. **Downstream compatibility** — NumPy, Pandas, SQLite, FAISS, and most Python libraries
   cannot operate on GPU tensors. They require CPU tensors (or NumPy arrays).
   Calling `.cpu()` before storing/comparing is mandatory.

---

## Cosine Similarity via Dot Product

Because embeddings are L2-normalized, comparing faces is a simple dot product:

```python
# Single comparison
score = torch.dot(emb1, emb2).item()  # returns float in [-1.0, 1.0]

# Batch comparison (find best match in database)
similarities = database_embs @ query_emb   # matrix-vector multiply: [N, 768] @ [768] → [N]
best_idx = similarities.argmax().item()
```

`.item()` extracts a Python scalar from a single-element tensor.
`.argmax()` returns the index of the highest similarity score.

---

## `torch.nn.functional` (F)

`F` contains *stateless* mathematical operations — they take inputs and return outputs
with no internal weights of their own (unlike `nn.Linear` or `nn.Conv2d`).

Common uses in this project:
- `F.normalize` — L2 normalization of embedding vectors
- Other ops available: `F.relu`, `F.softmax`, `F.cross_entropy`, `F.max_pool2d`

---

## Singleton Pattern

`FaceEmbedder` is designed to be a singleton — one shared instance rather than
reloading a 342 MB model on every call.

```python
_instance: "FaceEmbedder | None" = None
```

The module-level `_instance` variable is the placeholder for the single instance.
A factory function (not yet implemented) would check if `_instance` is `None`
before creating a new one.

**Why it matters here:** Loading DINOv3 is expensive (disk I/O + GPU transfer).
Creating multiple instances would waste memory and cause slow startup times.

---

## Device Selection Pattern

```python
def get_device():
    if torch.backends.mps.is_available():   # Apple Silicon (M1/M2/M3)
        return torch.device("mps")
    elif torch.cuda.is_available():          # NVIDIA GPU
        return torch.device("cuda")
    return torch.device("cpu")              # Fallback
```

MPS and CUDA both require `synchronize()` calls for accurate timing (relevant in
benchmarking scripts). For inference, `.to(device)` is all that's needed.

---

## Path Resolution

Always use `os.path.dirname(os.path.abspath(__file__))` to get the directory
of the *current file*, then build paths relative to it.

```python
cur_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_REPO = os.path.join(cur_dir, "dinov3")
```

This ensures the script works regardless of where it's called from.
Hardcoded paths or `os.getcwd()` break when scripts are run from different directories.
