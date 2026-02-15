'''
Docstring for visio_memoria.utils.dinov3_test
Purpose to benchmark dinov3 model of choice
Please refer to pytorch notes for more in depth detail on what each method or function does
'''

import os 
import torch
import time 
import numpy as np 
from transformers import AutoImageProcessor,  AutoModel
from torchvision.transforms import v2
from PIL import Image
import resource  # Unix system resource tracking

#IMPORTANT VARIABLES
IMG_DIM = 256


#===========================================================
#Configuration: Loading model 
#===========================================================
#LOADING MODEL AND WEIGHTS
curr_dir = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(curr_dir, "models", "dinov3")

#What is torch.hub.load 
#torch.hub.load is used to load the model from the local repository
#looks for hubconf.py file in the REPO_DIR
#source='local' means that the model is loaded from the local repository
print("Loading model...")
model = torch.hub.load(
    REPO_DIR,  #direction where 
    'dinov3_vitb16', 
    source='local', 
    weights='test' 
)


# What model.eval()?
# Neural networks behave differently during training vs inference.
# Some layers (like Dropout and BatchNorm) act randomly during
# training to prevent overfitting. eval() switches them to
# deterministic mode. Always call this before inference.
print("Setting model to evaluation mode...")
model.eval()

#Getting device to run model on 
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")



#===========================================================
#Transform Method: process image data 
#===========================================================

def make_transform(size:int  = IMG_DIM ) : 
    '''
    Docstring for make_transform
    
    :param size: Description
    :type size: int

    - DINOv3 expects images in a very specific format:
    - PyTorch tensor (not a PIL image or numpy array)
    - Shape: (batch, 3, height, width)  — 3 = RGB channels
    - Pixel values: floats, normalized with ImageNet stats
    - Raw images are usually (height, width, 3) with values 0-255.
    - The transform converts between these formats.

    '''
    return v2.Compose([
            
        # ToImage: converts PIL/numpy to a PyTorch tensor
        # Changes shape from (H, W, 3) to (3, H, W)
        # This is because PyTorch uses "channels first" format
        v2.ToImage(),

        #DINOv3 works with multiple of 16 (patch size)
        #resizes the image to a this size
        #antialias prevents jagged edges when downscaling
        v2.resize((size,size),  antialias = True),

        # ToDtype: converts pixel values from 0-255 integers
        # to 0.0-1.0 floats. Neural nets work with floats.
        # scale=True means divide by 255
        # values will be 0-1
        v2.ToDtype(torch.float32, scale=True),

        # Normalize: shifts and scales pixel values to match
        # what the model saw during training (ImageNet stats).
        # mean/std are per-channel (R, G, B).
        # Formula: pixel = (pixel - mean) / std
        # This is critical — wrong normalization = garbage embeddings
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])


transform = make_transform(256)


# Create a dummy image (random pixels, simulates a face crop)
# In real use, this would be a face crop from YOLOv8
dummy_pil = Image.fromarray(
    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
)

# Apply transform and add batch dimension

# WHAT IS unsqueeze(0)?
# The transform outputs shape (3, 256, 256) — one image.
# But PyTorch models expect a BATCH: (batch_size, 3, 256, 256)
# unsqueeze(0) adds a dimension at position 0:
#   (3, 256, 256) → (1, 3, 256, 256)
# This means "a batch of 1 image"
batch = transform(dummy_pil).unsqueeze(0)
print(f"Input shape: {batch.shape}")  # torch.Size([1, 3, 256, 256])


#===========================================================
#Benchmarking Functions 
#===========================================================

#BENCH MARK  1
def benchmark_latency(model, batch, device, num_warmup=10, num_runs=100):
    """
        Measure per-inference latency in milliseconds.
        How long does one inference take?

        Info about torch.inference_mode()
        - During Training Pytorch builds a computation graph
        - These generate a computation gradients used for training
        - We don't want that -> torch.inference_mode() disables that
        torch.no_grad(): Disables gradient calculation (good).
        - torch.inference_mode(): Disables gradient calculation AND skips view tracking and other overhead (best for latency). 
        - It is essentially a faster, specialized version of no_grad.
        
        Info about torch.sychronize 
        - Basically tell the code to wait up while a process finishes
        - MPS (and CUDA) run operations ASYNCHRONOUSLY — the Python code
        -you'd measure how long it takes to SUBMIT work, not FINISH it.
        - synchronize() blocks until the GPU is actually done.
    """
    
    model = model.to(device)
    batch = batch.to(device)
    
    # Warm up 
    # First few runs are always slower because:
    #   - MPS compiles Metal shaders on first use
    #   - Memory allocators are setting up pools
    #   - CPU caches are cold
    # We throw these away.
    print(f"  Warming up ({num_warmup} runs)...")
    with torch.inference_mode():

        for _ in range(num_warmup):

            _ = model(batch)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
    
    #Time Runs
    print(f"  Timing ({num_runs} runs)...")
    times = []
    with torch.inference_mode():

        for _ in range(num_runs):
            start = time.perf_counter()
            output = model(batch)

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000  # convert to ms
            times.append(elapsed)
    
    times = np.array(times)
    return {
        "mean_ms": times.mean(),
        "median_ms": np.median(times),
        "min_ms": times.min(),
        "max_ms": times.max(),
        "std_ms": times.std(),
        "fps": 1000 / times.mean(),
        "p95_ms": np.percentile(times, 95),  # 95th percentile — worst realistic case
        "p99_ms": np.percentile(times, 99),  # 99th percentile — near-worst case
    }


#BENCH MARK 2 
def benchmark_model_size(model):
    """
        Count parameters and estimate memory footprint.
        Parameters = the learned numbers inside the model.
        Each parameter is typically a float32 (4 bytes) or float16 (2 bytes).

        Info about .numel() and .nbytes()
        - .numel() = "number of elements" in a tensor
        - .nbytes = actual bytes in memory
        - .requires_grad tells you if this parameter would be
            - updated during training. For frozen inference, they're
              all True by default but we don't actually train.
        - model.parameters() is a generator that yields every weight
        - tensor in the model (convolution filters, attention matrices, etc.)
    """
    
    total_params = 0
    total_bytes = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        total_bytes += param.nbytes
        
       
        if param.requires_grad:
            trainable_params += param.requires_grad
    
    return {
        "total_params": total_params,
        "total_params_M": total_params / 1e6,
        "model_size_MB": total_bytes / 1e6,
        "fp16_size_MB": (total_params * 2) / 1e6,  # 2 bytes per float16
    }


#BENCHMARK #3 Memory Usuage
def get_process_memory() -> tuple:
    '''
    - get current process memory usage from the OS.
    - returns tuple of (mb , b )

    '''

    # ru_maxrss is "max resident set size" — peak physical RAM used
    # On macOS it's in bytes, on Linux it's in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF)
    size_in_mb = usage.ru_maxrss / (1024 * 1024)  # convert to MB
    size_in_b = usage.ru_maxrss 

    return (size_in_mb, size_in_mb)




#-----