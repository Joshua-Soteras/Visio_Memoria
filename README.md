# Visio_Memoria

Facial recognition system that detects, remembers, and greets people it has seen before. 



---

Visio_Memoria 

## Tech Stack


--- 


## System Pipeline 

```
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

```


--- 


### Active python env 
source .venv/bin/activate

--- 

## Learning YOLO 

- assuming model = YOLO(load_model)
- What is returned from model() is a Results object
- model can return a list of results 

- .boxes	
       - Boxes Object	
       - (Most Important) Contains bounding box coordinates, confidence scores, and class IDs. Used in Object Detection & Instance Segmentation.
- .keypoints	
       - Keypoints Object	
       - Contains x, y coordinates (and visibility) for specific landmarks. 
       - Crucial for  face model (eyes, nose, mouth).
- .masks	
       - Masks Object	
       - Contains segmentation masks (pixel-level outlines). 
       - None for your face model unless it is a segmentation model.
- .probs	
       - Probs Object	
       - Contains classification probabilities (e.g., "99% chance this whole image is a cat"). - Used only in Classification tasks.
- .obb	OBB Object	
        - Oriented Bounding Boxes (rotated boxes), 
        - used for aerial imagery or  angled objects.


```import import numpy as np``` 


```frame: np.ndarray``` 
- Canvas 
- the full-sized image / single frame of vidio that was fed into yolo to get predictions
- 3D matrix of numbers representing pixels (height x widith x colors)


```bbox```
- bounding box
- 


--- 


##  Learning CV / Pytorch 
- pixel coordinates must be whole numbers

### Using Torchvision 

```python
from torchvision.transform import v2
```

- this is for image processing 
- converting the image to a tensor for pytorch to understand
- transforms.v1 is legacy 


Example Usage: 

```python
TRANSFORM = v2.Compose([
       v2.ToImage(),
       v2.Resize((size,size) , antialias = True).
       v2.ToDType(float32, scale = True )
       v2.Normalize(mean =(), std )
])
```

- ToImage
       - converts PIL/numpy to a PyTorch tensor
       - Changes shape from (H, W, 3) to (3, H, W)
       - This is because PyTorch uses "channels first" format

-  Resize()
       - DINOv3 works with multiple of 16 (patch size)
       - resizes the image to a this size
       - antialias prevents jagged edges when downscaling

- ToDtype: converts pixel values from 0-255 integers
       - to 0.0-1.0 floats. Neural nets work with floats.
       - scale=True means divide by 255
       - values will be 0-1

- Normalize: 
       - shifts and scales pixel values to match
       - what the model saw during training (ImageNet stats).
       - mean/std are per-channel (R, G, B).
       - Formula: pixel = (pixel - mean) / std
       - This is critical — wrong normalization = garbage embeddings


### Loading a Model 

```python
torch.hub.load()
```

- loads a specific model 
- looks for the hubconf.py file in REPO_DIR

Example Usage: 

```python
model = torch.hub.load(
       REPO_DIR,  #direction where
       'dinov3_vitb16',
       source='local',
       weights=WEIGHTS
       )
```


### .eval()
- Neural networks behave differently during training vs inference.
       - Some layers (like Dropout and BatchNorm) act randomly during
       - training to prevent overfitting. eval() switches them to
       - deterministic mode. Always call this before inference.

 ```python 
 model.eval()
 ```


### with torch.inference_mode( )
- Telling pytorch you will not call .backward()
- Disables Gradient Calculation (Autograd)
       - Pytorch stops tracking operations to build the computational graph
       - saves significant memory by not storing back propagation
       - Disables version tracking

       
### inference_mode vs. no_grad vs. eval

```model.eval()```
- Changes layer behavior. It tells layers like Dropout and BatchNorm to switch to testing mode (e.g., disable dropout).

```torch.no_grad()```
- Always call this before inference.
- Disables gradient calculation.
- Useful, but inference_mode is now preferred.

```torch.inference_mode()```
- Disables gradients AND disables version tracking. It is the "extreme" optimization mode.
- Recommended for pure inference (prediction) where you never need to train.


--- 


## Python 

### Singletons:
- A design pattern that restics a class to having only one instance while providing global access
- Created for resource management and performance 

Strict Resource Cotnrol
- One at a time rule
- prevents corruption 
- 

Drawbacks
- Anti-pattern 


### Exception Handling 
```python
try: 
       #code goes here
except Exception as e:
       print(#print errors here )
       exit()

``` 
- e is the error.
- e is a variable name that stores the error 
- usually exit() is placed in the excepttion block 



---

### References
- https://github.com/derronqi/yolov8-face?tab=readme-ov-file
- https://github.com/facebookresearch/dinov3?tab=readme-ov-file
