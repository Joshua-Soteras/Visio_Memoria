"""
Docstring for dinov3_bm
Purpose: benchmark dinov3 model and to see if model is running correctly 
"""


import torch 
import platform
import psutil
from transformers import AutoImageProcessor, AutoModel
from PIL import Image 
import os 


#Check System specs 

def get_size(bytes, suffix ="B"): 
    """
    Docstring for get_size
    used to get size of GB for 
    """

    #base 10 for binary 2^10
    factor = 1024

    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def print_system_specs(): 

    """
    Docstring for print_system_specs
    Displaying hardware specs of user's pc
    """
    #1.CPU INFO
    # Knowing your OS is important because some libraries (like bitsandbytes)
    # have different installation commands for Windows vs. Linux.
    print("\nOS and CPU INFO")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}") # e.g., 'x86_64' or 'arm64' (Apple Silicon)
    print(f"Processor: {platform.processor()}")

    # "Physical cores" are the actual hardware units.
    # "Total cores" includes "Hyper-threading" (virtual cores).
    # AI data loading (DataLoader) works best when num_workers <= Physical Cores.
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total cores:    {psutil.cpu_count(logical=True)}")


    #2.RAM INFO
    # Systems main memory
    # if model is too big for GPU VRAM, can offload here at cost of efficiency 
    # same thing for cpu 
    print("\nRAM INFO")
    svmem = psutil.virtual_memory()
    print(f"Total RAM:      {get_size(svmem.total)}")
    print(f"Available RAM:  {get_size(svmem.available)}")


    #3.GPU Info (The most important part) 
    print("\nGPU / Accelerator Status:")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available (NVIDIA GPU detected)")
        device_count = torch.cuda.device_count()
        print(f"   GPU Count: {device_count}")
        
        for i in range(device_count):

            #name of gpu 
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)

            # Total VRAM: The maximum model size you can fit.
            # Example: A 7B parameter model needs ~14GB VRAM (in float16).
            total_vram = gpu_props.total_memory
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            free_vram = total_vram - reserved
            
            print(f"   GPU {i}: {gpu_name}")
            print(f"     - Total VRAM:     {get_size(total_vram)}")
            print(f"     - Reserved VRAM:  {get_size(reserved)} (Used by PyTorch cache)")
            print(f"     - Allocated VRAM: {get_size(allocated)} (Used by Tensors)")
    
    # CASE B: Apple Silicon (MPS)
    # Checks for M1/M2/M3 chips.
    elif torch.backends.mps.is_available():
        print(f"\t MPS is available (Apple Silicon)")
        print(f"\t Device: Apple M-Series Chip")
        
        # Crucial Note for Mac Users:
        # Macs don't have separate VRAM. They have "Unified Memory".
        # If you have 16GB Total RAM, your GPU also has access to ~11-12GB of it.
        print(f"\t Note: On Mac, VRAM is shared with System RAM (Unified Memory).")
        print(f"\tCheck 'Available RAM' above for your effective limit.")
        
    # CASE C: No Accelerator
    # If you see this, training will be impossibly slow, but inference (running once) might work.
    else:
        print("No GPU accelerator detected.")
        print("Models will run on CPU. This will be slow for DINOv3.")
    
    print("\n")

   
print_system_specs()

#Dynamic Pathing 
#gets the directory where this script exist 
current_dir = os.path.dirname(os.path.abspath(__file__))

#construct the path dyanamically 
model_path = os.path.join(current_dir, "dinov3")
