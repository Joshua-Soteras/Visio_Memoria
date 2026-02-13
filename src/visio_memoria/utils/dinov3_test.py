import os 
import torch
import time 
import numpy as np 
from transformers import AutoImageProcessor,  AutoModel
from PIL import Image


#Config---
curr_dir = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(curr_dir, "models", "dinov3")

#loading model
dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)