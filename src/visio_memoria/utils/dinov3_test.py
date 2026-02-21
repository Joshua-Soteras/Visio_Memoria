'''
Docstring for visio_memoria.utils.dinov3_test
Purpose to benchmark dinov3 model of choice
Please refer to pytorch notes for more in depth detail on what each method or function does
'''

import os
import torch
import time
import numpy as np
from torchvision.transforms import v2
from PIL import Image
import resource  # Unix system resource tracking
from dotenv import load_dotenv


#IMPORTANT VARIABLES
IMG_DIM = 256

#===========================================================
#Configuration: Path constants
#===========================================================
load_dotenv() 
WEIGHTS = os.getenv("weights_vitb16")
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
REPO_DIR = os.path.join(parent_dir, "models", "dinov3")
WEIGHTS = os.path.join(parent_dir, "models" , WEIGHTS)
IMAGE_SAMPLE = os.path.join(curr_dir, "IMG_SAMPLE.jpg")



#Getting device to run model on
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


#===========================================================
#Transform Method: process image data
#===========================================================

def make_transform(size:int = IMG_DIM ) :
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
        v2.Resize((size,size),  antialias = True),

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


def get_batch_sample(img:Image = IMAGE_SAMPLE):
    '''
    Create a dummy image (random pixels, simulates a face crop).
    In real use, this would be a face crop from YOLOv8.
    Returns a batch tensor of shape (1, 3, IMG_DIM, IMG_DIM).
    '''

    #1. Load a real image file
    # Replace 'my_image.jpg' with the actual path to your file

    # .convert("RGB") ensures the image has 3 channels (Red, Green, Blue)
    # This prevents errors if your image is Grayscale (1 channel) or RGBA (4 channels)
    real_image_pil = Image.open(img).convert("RGB") 

    
    # Apply transform and add batch dimension
    # The transform outputs shape (3, H, W) — one image.
    # But PyTorch models expect a BATCH: (batch_size, 3, H, W)
    # unsqueeze(0) adds a dimension at position 0:
    # (3, H, W) → (1, 3, H, W)
    # This means "a batch of 1 image"
    # Assuming 'transform' is already defined in your code (e.g., Resize, ToTensor)
    transform = make_transform()
    pil_img = Image.open(img).convert("RGB")
    batch = transform(real_image_pil).unsqueeze(0)
    print(f"Batch Sample - Input shape: {batch.shape}")  # torch.Size([1, 3, 256, 256])
    return batch 

    '''
    Creating Psuedo Image 
    transform = make_transform()

    dummy_pil = Image.fromarray(
        np.random.randint(0, 255, (IMG_DIM, IMG_DIM, 3), dtype=np.uint8)
    )
    # WHAT IS unsqueeze(0)?
    # The transform outputs shape (3, 256, 256) — one image.
    # But PyTorch models expect a BATCH: (batch_size, 3, 256, 256)
    # unsqueeze(0) adds a dimension at position 0:
    #   (3, 256, 256) → (1, 3, 256, 256)
    # This means "a batch of 1 image"
    batch = transform(dummy_pil).unsqueeze(0)
    return batch
    '''



#===========================================================
#BENCH MARK  1
#===========================================================
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



#===========================================================
#BENCH MARK 2
#===========================================================
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



#===========================================================
#BENCHMARK #3 Memory Usuage
#===========================================================
def get_process_memory() -> tuple:
    '''
    - get current process memory usage from the OS.
    - returns tuple of (mb , b )
    -ru_maxrss is "max resident set size" — peak physical RAM used
    - On macOS it's in bytes, on Linux it's in kilobytes
    '''

    usage = resource.getrusage(resource.RUSAGE_SELF)
    size_in_mb = usage.ru_maxrss / (1024 * 1024)  # convert to MB
    size_in_b = usage.ru_maxrss

    return (size_in_mb, size_in_b)


def benchmark_memory(model, batch, device):
    """Measure memory consumption at each stage."""

    mem_before_load_mb, mem_before_load_b = get_process_memory()

    model = model.to(device)
    mem_after_model_mb, mem_after_model_b = get_process_memory()

    # Input must also be on the same device
    batch = batch.to(device)
    # The image tensor gets copied to the same place.
    # If model is on MPS and batch is on CPU → crash:
    #   "Expected all tensors to be on the same device"

    with torch.inference_mode():
        output = model(batch)
        if device.type == "mps":
            torch.mps.synchronize()

    mem_after_inference_mb, mem_after_inference_b= get_process_memory()

    # MPS-specific memory tracking (if available)
    mps_allocated = None
    if device.type == "mps":
        try:
            # .current_allocated_memory() returns bytes currently
            # held by MPS. This is the GPU-side memory usage.
            mps_allocated = torch.mps.current_allocated_memory() / 1e6
        except AttributeError:
            pass  # older PyTorch versions don't have this

    return {

        " \n - Before Model: Running Process Memory (MB) ": mem_before_load_mb,
        " - Before Model: Running Process Memory (B) ": mem_before_load_b,

        " \n -After Model: Running Process Memory (mb) ": mem_after_model_mb,
        " - After Model: Running Process Memory (b) ": mem_after_model_b,

        "\n - model_load_cost_MB": mem_after_model_mb - mem_before_load_mb,
        " - inference_cost_MB": mem_after_inference_mb - mem_after_model_mb,
        "mps_allocated_MB": mps_allocated,
    }



# ============================================================
# BENCHMARK 4: BATCH SIZE SCALING
# ============================================================
# If multiple faces appear in one frame, you can either:
#   - Process them one by one (batch_size=1, simple)
#   - Stack them into a batch (batch_size=N, potentially faster)
#
# GPUs are parallel processors — they can handle multiple images
# at once. But there's overhead in batching, and memory grows
# linearly. This benchmark finds the sweet spot.
# ============================================================

def benchmark_batch_sizes(model, device, batch_sizes=[1, 2, 4, 8, 16]):
    """Measure latency and per-image throughput at different batch sizes."""
    
    model = model.to(device)
    results = {}
    
    for bs in batch_sizes:
        # Create a batch of bs identical images
        dummy = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        single = make_transform(256)(dummy)
        
        # .repeat() tiles the tensor along dimensions
        # (1,1,1) means: 1x along channels, 1x along height, 1x along width
        # We use .unsqueeze(0).repeat(bs,1,1,1) to make a batch
        batch = single.unsqueeze(0).repeat(bs, 1, 1, 1).to(device)
        
        with torch.inference_mode():
            for _ in range(5):
                _ = model(batch)
                if device.type == "mps":
                    torch.mps.synchronize()
            
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = model(batch)
                if device.type == "mps":
                    torch.mps.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        
        mean_ms = np.mean(times)
        per_image_ms = mean_ms / bs
        
        results[bs] = {
            "total_ms": mean_ms,
            "per_image_ms": per_image_ms,
            "throughput_fps": (bs / mean_ms) * 1000,
        }
        print(f"  batch={bs}: {mean_ms:.1f}ms total, {per_image_ms:.1f}ms/image, {(bs/mean_ms)*1000:.0f} FPS")
    
    return results



#===========================================================
# Main Entry Point
#===========================================================
def main():

    #--- Validate sample image ---
    # 1. EXISTENCE CHECK
    # Does the file actually sit on the hard drive?
    print("\n--- Configuration: Loading Sample Image ---")
    if os.path.exists(IMAGE_SAMPLE):
        print(f"Path found: {IMAGE_SAMPLE}")
    else:
        print(f"ERROR (Image Loading): File not found at {IMAGE_SAMPLE}")
        print(f"   -> Current Working Directory is: {os.getcwd()}")
        # Stop execution here if file is missing to avoid crashing later
        return

    # 2. LOAD CHECK
    # Can PIL actually read the bytes? (Detects corrupted files)
    print("--- Config: PIL reading Image ---")
    try:
        img = Image.open(IMAGE_SAMPLE)
        print("✅ PIL opened the image successfully.")

        # 3. METADATA CHECK
        # - Size: (Width, Height). DINOv3 usually needs square inputs eventually.
        # - Mode: 'RGB' is good. 'RGBA' (transparent) or 'L' (grayscale) might need conversion.
        print(f"   - Dimensions: {img.size} (Width x Height)")
        print(f"   - Format: {img.format}")  # e.g., JPEG, PNG
        print(f"   - Mode: {img.mode}")      # e.g., RGB, RGBA

        # 4. DATA CHECK (The "Is it empty?" Test)
        # We convert to a numpy array to look at the raw pixel numbers.
        # This ensures the image isn't just a solid black square.
        img_array = np.array(img)

        print(f"Pixel Data Loaded.")
        print(f"   - Shape: {img_array.shape}") # Should be (H, W, 3) for RGB
        print(f"   - Min Value: {img_array.min()} (Should be >= 0)")
        print(f"   - Max Value: {img_array.max()} (Should be <= 255)")
        print(f"   - Mean Intensity: {img_array.mean():.2f}")

        if img_array.max() == 0:
            print("WARNING: Image is completely black (all zeros).")
        elif img_array.std() < 1:
            print(" WARNING: Image is a single solid color (std dev ~ 0).")
        else:
            print("Image has variation (it looks like a real picture).")

        print ("\n")

    except Exception as e:
        print(f"Failed to load image data: {e} \n")
        return

    #--- Load model ---
    # What is torch.hub.load
    # torch.hub.load is used to load the model from the local repository
    # looks for hubconf.py file in the REPO_DIR
    # source='local' means that the model is loaded from the local repository
    # 2. .to(device) — copies weights to GPU memory space
    #model = model.to(device)
    # If device is "mps", PyTorch copies every tensor to the
    # Metal GPU's address space. On NVIDIA this would be VRAM.
    # On M4 it's the same physical RAM chip but a different
    # allocation pool managed by Metal
    print("--- Config: Loading Model and Mode ---")
    try: 
        model = torch.hub.load(
            REPO_DIR,  #direction where
            'dinov3_vitb16',
            source='local',
            weights=WEIGHTS
        )
        print("Model has Successfully Loaded\n")
    except Exception as e: 
        print("ERROR (Loading Model: Could not load model correctly\n)")
        exit()

    
    # What model.eval()?
    # Neural networks behave differently during training vs inference.
    # Some layers (like Dropout and BatchNorm) act randomly during
    # training to prevent overfitting. eval() switches them to
    # deterministic mode. Always call this before inference.
    print("Setting model to evaluation mode...")
    model.eval()

    device = get_device()
    print(f"Using device: {device}")


    #--- Create batch ---
    print("--- Config: Batch Sample Information---")
    batch = get_batch_sample()

    
    #================================================================
    #--- Run benchmarks ---
    print("\n--- Benchmark: Latency ---")
    latency = benchmark_latency(model, batch, device)
    for k, v in latency.items():
        print(f"  {k}: {v:.2f}")

    print("\n--- Benchmark: Model Size ---")
    size = benchmark_model_size(model)
    for k, v in size.items():
        print(f"  {k}: {v:.2f}")

    print("\n--- Benchmark: Memory ---")
    mem = benchmark_memory(model, batch, device)
    for k, v in mem.items():
        print(f"  {k}: {v}")

    print("\n--- Benchmark: Batch Size Scaling ---")
    print(benchmark_batch_sizes(model, device))

    exit()

if __name__ == "__main__":
    main()
