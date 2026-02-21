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

```python
cap = cv2.VideoCapture(0) # Try 0, then 1 if this fails
success , frame = cap.read()
```

- cap is just a variable that you have to ask to fetch a frame
- success:bool -> did I capture an image 
- frame: the acuall numpy array (the image )
- cap.reaD() method call to capture the frame 

```frame: np.ndarray``` 
- Canvas 
- the full-sized image / single frame of vidio that was fed into yolo to get predictions
- 3D matrix of numbers representing pixels (height x widith x colors)


Example of loop of detecting faces
Paying attentioin to what is stored in 
```
results 
```
when frame in passed into the model

```python
while True:
    ret, frame = cap.read()
    if not ret:
       break
    
    results = model(frame, conf=0.4, verbose=False)
    annotated = results[0].plot()
    
    cv2.imshow("Visio_Memoroia", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```
- results[0]
       - Inside results[0], there is the .boxes attribute.
       - If there is 1 face, .boxes holds 1 set of coordinates.
       - If there are 3 faces, .boxes holds 3 sets of coordinates stacked on top of each other.
       - If you printed out the raw math (results[0].boxes.xyxy), it would look like a 2D grid (a matrix) instead of a single line
- results[1]
       - conf
       -How sure the AI is that it actually found the object (0.0 to 1.0).
- results[2]
       - A number representing what it found (e.g., 0 = person, 1 = car, or in your case, face).



```bbox```
- bounding box
- Coordinates of the box that detects/classifies 
- List of array of 4 numbers
- ```[x1,y1,x2,y2]``` 
       - x1: The pixel coordinate of the Left edge.
       - y1: The pixel coordinate of the Top edge.
       - x2: The pixel coordinate of the Right edge.
       - y2: The pixel coordinate of the Bottom edge.


## from PIL import Image
```
#Crop and convert BGR → RGB → PIL 
face_crop = frame[y1:y2 , x1:x2]
face_rgb = face_crop[:,:,::-1]

'''
returns a PIL objects 
from raw numbers to fully formed image 
why: any modern ai libraries (like pytorch facenet, or clip) reuqure PIL Images
'''
return Image.fromarray(face_rgb)
```
- forming numbers to pil 

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

<b>torchvision.transforms.v2.compose</b>
```python
TRANSFORM = v2.Compose([
       v2.ToImage(),
       v2.Resize((size,size) , antialias = True).
       v2.ToDType(float32, scale = True )
       v2.Normalize(mean =(), std )
])
```

- v2.compose([])
       - comes from torchvision,.transforms.v2 
       - does not return an image
       - a callable (acts like a function)

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

<b>.unsqueeze()</b>
```
transform = TRANSFORM()
pil_img = Image.open(img).convert("RGB")
batch = transform(real_image_pil).unsqueeze(0)
```

- Image.open(img)
       - uses Pillow (Pil) to load the image file 
       - .convert("RGB")
       - crucial step : some images are black and white 
       - ai model usally expect 3 color channles 
- transform
       - changes shape of image 
       - see previous function 
- .unsqueeze(0)
       - deep learning models are designed to process multiple images at once
       - this is called a batch
       - because of this, the model strictly refuses to accept a single 3D image 
              - [Channels,Height, Width] -> single image 
              - 4D input: [Batch_Size, Channels, Height, Width]
       - .unsqueeze(0) tells PyTorch: 
              - "Take my single 3D image, and wrap it inside a new 1st dimension (dimension 0) so it looks like a list of 1 image



### torch.nn.functional

<b>functional </b>

- It is a module within PyTorch that contains purely functional (stateless) operations. - Unlike standard neural network layers (like nn.Linear or nn.Conv2d which store their own learnable weights and biases) 
- the functions inside torch.nn.functional just take inputs, perform a mathematical operation, and return an output.

- It contains functions for:
       - Normalization: F.normalize
       - ctivation functions: F.relu, F.sigmoid, F.softmax
       - Loss functions: F.cross_entropy, F.mse_loss
       - ooling operations: F.max_pool2d
       - So when your code calls F.normalize(...), it is reaching into that functional 
       - module to run the L2 normalization math on the tensor you passed to it.



<b>.normalize() method</b>

```python
def get_embedding(self, face_image: Image.Image) -> torch.Tensor:
        """
        Extract an L2-normalized embedding from a face crop.

        Args:
            face_image: PIL Image of a cropped face

        Returns:
            Normalized embedding tensor, shape (embed_dim,)
        """
        # Preprocess: resize, to tensor, normalize
        tensor = self.TRANSFORM(face_image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.inference_mode():
            embedding = self.model(tensor)

        # L2 normalize so cosine similarity = dot product
        embedding = F.normalize(embedding[0], p=2, dim=0)
        return embedding.cpu()
```

Breakdown of the Arguments

-In your code: F.normalize(embedding[0], p=2, dim=0)

-embedding[0] (The Input):
-Earlier in the code, unsqueeze(0) was used to add a batch dimension, turning your single image into a "batch of 1". 
       - The model outputs a tensor with the shape (1, embed_dim). By using embedding[0], you strip away that batch dimension, grabbing the actual 1D vector of shape (embed_dim,) to normalize.
-p=2 (The Norm Degree)
       - This tells PyTorch to use the L2 norm (Euclidean norm). 
       - If it were p=1, it would use the L1 norm (Manhattan distance, summing the absolute values). 
       - p=2 squares the values, sums them, and takes the square root to find the length.
- dim=0 (
       - The Dimension): This tells PyTorch which axis to calculate the length across
       - Because you already extracted embedding[0], you are working with a flat 1D tensor. 
       - Therefore, dim=0 is the only dimension available (the elements of the vector itself).

       The .cpu() method moves the tensor from its current processing device (like a graphics card) back into your computer's standard system memory (CPU RAM).

<b>Importance or return embedding.cpui </b>
Earlier in your code, tensor = ...to(self.device) likely moved the image data onto a GPU to make the neural network's forward pass run significantly faster. Calling .cpu() at the very end simply brings the final result back.

Here is why this is a crucial step in machine learning pipelines:

1. Saving Precious GPU Memory (VRAM)
- GPU memory is expensive and highly limited compared to standard system RAM. 
- For instance, if self.device is routing this work to your RTX 4070, you are working with a finite amount of VRAM. - If you were extracting embeddings for thousands of faces and leaving them all on the GPU, you would quickly crash your program with a CUDA "Out of Memory" (OOM) error.
- By pushing the tiny 1D embedding vector back to the CPU, you immediately free up that VRAM space for the next heavy image-processing task, while safely storing the lightweight result in your system's much larger pool of RAM.

2. Downstream Compatibility
- Most of the tools you will use to store or compare these embeddings do not know how to interact with GPU memory. If you want to:
- Save the embedding to a database (like PostgreSQL or MongoDB)
- Convert it to a standard Python list
- Use standard data science libraries like NumPy, Pandas, or Scikit-Learn
- Pass it to a frontend application (like your Nuxt dashboard)
- ...the data almost always needs to be on the CPU first. If you try to run a NumPy operation on a tensor that is still living on the GPU, PyTorch will immediately throw an error.


<b>batch vs layer normalizaiton  </b>
- Batch normalization 
       - Normalizes features across the batch dimension. If you have a batch of 32 images, it calculates the mean and variance of a specific feature across all 32 images.
- Layer Normailiztion 
       -Normalizes features across the feature dimension for a single data point. It calculates the mean and variance across all features of one specific image or token.



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

       
<b> inference_mode vs. no_grad vs. eval</b> 

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
