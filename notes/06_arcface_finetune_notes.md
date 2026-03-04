# Fine-Tuning DINOv3 with ArcFace Loss — Concepts & Implementation

Notes on how to fine-tune DINOv3 with ArcFace loss for pose-invariant face recognition.
Each section maps to something in the implementation.

---

## How To Learn This From Scratch

If you had no reference, here is the exact order you would study this:

```
Step 1 → Understand WHY normal classification fails for faces
Step 2 → Learn what an embedding space is
Step 3 → Learn cosine similarity and angular distance
Step 4 → Learn what ArcFace loss does differently
Step 5 → Learn transfer learning (frozen backbone + projection head)
Step 6 → Put it all together in a training loop
```

---

## Step 1 — Why Normal Classification Fails for Faces

A standard classifier learns:

```
image → model → "Is this Alex? Is this Josh? Is this Sara?"
         ↓
   [0.9, 0.05, 0.05]   ← probability over known people
```

**The problem:**
- Works only for people it was trained on
- You add a new person → retrain the entire model
- Doesn't generalise to new identities at runtime

What you actually want:

```
image → model → a number vector (embedding)
                    ↓
             compare vectors to find who is closest
```

This is called **metric learning** — instead of classifying, you learn a space
where the same person's face always lands near itself, regardless of angle.

---

## Step 2 — What Is an Embedding Space

An embedding is just a list of numbers (a vector) that represents an image.

```
Face image (256×256 pixels)
    ↓
DINOv3 ViT-B/16
    ↓
[0.12, -0.43, 0.87, 0.02, ... ]   ← 768 numbers
```

The goal: two photos of the same person should produce vectors that are **close**.
Two different people should produce vectors that are **far apart**.

```
GOOD embedding space:

        Alex_front ●  ● Alex_left
                    ●
                 Alex_right

                              Josh_front ●  ● Josh_left


BAD embedding space (what DINOv3 gives you for faces):

   Alex_front ●

                     Alex_left ●

   Josh_front ●

                               Alex_right ●
                     Josh_left ●
```

ArcFace training teaches the model to produce the GOOD version.

---

## Step 3 — Cosine Similarity and Angular Distance

To compare two embeddings, you measure the **angle** between them, not the
raw distance. This is called cosine similarity.

```
Cosine similarity formula:

         A · B
cos θ = ───────
         |A||B|

- Result is between -1 and 1
- cos θ = 1   → same direction → same person
- cos θ = 0   → perpendicular → unrelated
- cos θ = -1  → opposite      → very different
```

Why angle, not distance?

```
● Alex_front (length 5)         ● Alex_right (length 2)

Both point in the same DIRECTION even though they have different lengths.
L2 normalization makes all vectors length 1 so only the angle matters.

After L2 normalization:
         ↗ Alex_front (length 1)
         ↗ Alex_right (length 1)   ← same direction = same person
```

In PyTorch:
```python
embedding = F.normalize(embedding, p=2, dim=1)   # makes length = 1
cosine = F.linear(embedding, weight)              # dot product = cosine similarity
```

---

## Step 4 — What ArcFace Loss Does

### Regular Softmax (what normal classification uses)

```
cosine similarity → scale up → softmax → cross entropy loss
```

The model just needs to get the correct class score higher than others.
It can do this lazily — embeddings don't need to be tightly clustered.

### ArcFace (what we use)

```
cosine similarity → convert to angle → ADD MARGIN to correct class → cos back → scale → cross entropy
```

The extra step: **add a penalty angle (m) to the correct class before computing loss**.

```
Without margin:

    Alex class center
           ↑
           |  θ = 20°   ← easy, model is comfortable here
           |
    Alex embedding

With margin (m = 0.5 rad ≈ 28°):

    Alex class center
           ↑
           |  θ + m = 48°  ← now the model has to work harder to get this right
           |
    Alex embedding
```

By forcing the model to push embeddings even closer to their class center than
it "needs" to, you get much tighter, better-separated clusters.

```
ArcFace loss diagram:

Identity centers (learned):

        Alex ●          Josh ●          Sara ●

                 embedding space

During training, ArcFace pulls each embedding toward its center
while the margin forces it to be VERY close, not just close enough.

Result after training:

       Alex_front ●
       Alex_left  ●  ← all very tight around Alex center
       Alex_right ●

                              Josh_front ●
                              Josh_left  ●  ← tight around Josh center
```

---

## Step 5 — Transfer Learning: Frozen Backbone + Projection Head

You do not train DINOv3 from scratch. It already knows how to see.
You only teach it to organise faces better.

```
WHAT IS FROZEN vs TRAINABLE:

┌─────────────────────────────────┐
│         DINOv3 ViT-B/16        │  ← FROZEN
│   (pretrained on ImageNet)      │     no gradients flow here
│   outputs 768-d feature vector  │     weights don't change
└────────────────┬────────────────┘
                 │ 768-d features
                 ▼
┌─────────────────────────────────┐
│       Projection Head           │  ← TRAINABLE
│   Linear(768 → 512)             │     learns from ArcFace loss
│   BatchNorm1d(512)              │     weights update every step
└────────────────┬────────────────┘
                 │ 512-d face embedding
                 ▼
┌─────────────────────────────────┐
│         ArcFace Loss            │  ← TRAINABLE
│   class weight matrix           │     one vector per identity
│   (num_identities × 512)        │     updates alongside projection head
└─────────────────────────────────┘
```

Why freeze DINOv3?
- It took months and massive compute to train
- Its features are already very good
- You just need to re-map them into face-specific space
- Training the full model requires far more data and GPU hours

Why BatchNorm in the projection head?
- Stabilises training by keeping embedding values in a consistent range
- Prevents any single dimension from dominating the cosine similarity

---

## Step 6 — The Training Loop (What Actually Happens)

```
ONE TRAINING STEP:

1. Load a batch of face images + their identity labels
         [Alex_img, Josh_img, Alex_img, Sara_img]
         [    0,       1,        0,       2    ]   ← labels as integers

2. Forward pass through frozen DINOv3
         768-d features (no gradients tracked)

3. Forward pass through projection head
         512-d embeddings (gradients tracked here)

4. ArcFace loss computes:
    a. L2 normalize embeddings and class weight vectors
    b. Cosine similarity between each embedding and all class centers
    c. Convert cosine → angle (acos)
    d. Add margin to correct class angle
    e. Convert back to cosine
    f. Scale by s=64
    g. Cross entropy loss

5. loss.backward()
    → gradients flow back through projection head + ArcFace weights
    → DINOv3 gets NO gradients (frozen)

6. optimizer.step()
    → only projection head + ArcFace class centers update

7. Repeat for all batches → one epoch done
```

Gradient flow diagram:

```
ArcFace loss
     ↑  gradients flow up
Projection head   (weights update ✓)
     ↑  gradients flow up
DINOv3 backbone   (torch.no_grad() blocks gradients ✗)
     ↑  no gradients
Input image
```

---

## Dataset Structure Required

```
data/
└── faces/
    ├── alex/
    │   ├── front.jpg
    │   ├── left.jpg
    │   └── right.jpg
    ├── josh/
    │   ├── front.jpg
    │   └── angled.jpg
    └── sara/
        ├── front.jpg
        └── tilted.jpg
```

- Each subfolder = one identity
- Minimum: ~5-10 images per person, different angles
- Better: 20+ images per person
- At scale: use VGGFace2 (3M images, 9K identities)

---

## Full Implementation

### ArcFace Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    Adds a fixed angular penalty (m) to the correct class angle before softmax.
    Forces embeddings to cluster tightly in angular space.
    Paper: https://arxiv.org/abs/1801.07698
    """
    def __init__(self, embed_dim: int, num_identities: int, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale      # s: amplifies cosine logits (usually 64)
        self.margin = margin    # m: angular penalty in radians (usually 0.5)

        # one learnable weight vector per identity — the "class centers"
        self.weight = nn.Parameter(torch.FloatTensor(num_identities, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # L2 normalize so dot product = cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)       # (B, embed_dim)
        weight     = F.normalize(self.weight,    p=2, dim=1)   # (num_ids, embed_dim)

        # cosine similarity between each embedding and each class center
        cosine = F.linear(embeddings, weight).clamp(-1 + 1e-7, 1 - 1e-7)  # (B, num_ids)

        # convert cosine → angle
        theta = torch.acos(cosine)

        # add margin ONLY to the correct class angle
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # penalised angle → back to cosine → scale → cross entropy
        output = torch.cos(theta + self.margin * one_hot) * self.scale
        return F.cross_entropy(output, labels)
```

### Model

```python
class DINOv3FaceModel(nn.Module):
    """
    Frozen DINOv3 backbone + trainable projection head.
    Only the projection head learns face-specific geometry.
    """
    def __init__(self, backbone: nn.Module, embed_dim: int = 512):
        super().__init__()

        # freeze all DINOv3 parameters
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # trainable head: re-maps 768-d features → 512-d face embeddings
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)      # (B, 768) — no gradients
        return self.projection(features)     # (B, 512) — gradients flow here
```

### Dataset

```python
import os
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    """
    Loads faces from a folder-per-identity structure.
    Automatically assigns integer labels from folder names.
    """
    def __init__(self, root_dir: str, transform):
        self.transform = transform
        self.samples = []        # (image_path, label)
        self.identity_map = {}   # name → int

        for label_idx, identity in enumerate(sorted(os.listdir(root_dir))):
            identity_dir = os.path.join(root_dir, identity)
            if not os.path.isdir(identity_dir):
                continue
            self.identity_map[identity] = label_idx
            for img_file in os.listdir(identity_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(identity_dir, img_file), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label
```

### Training

```python
from torchvision.transforms import v2
from torch.utils.data import DataLoader

def train():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_DIR   = os.path.join(SCRIPT_DIR, "../models/dinov3")
    WEIGHTS    = os.path.join(REPO_DIR, "dinov3_vitb16_pretrain.pth")
    DATA_DIR   = os.path.join(SCRIPT_DIR, "../data/faces")
    SAVE_PATH  = os.path.join(SCRIPT_DIR, "../models/dinov3_arcface_head.pth")

    # device: MPS → CUDA → CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load frozen DINOv3 backbone
    backbone = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=WEIGHTS)
    backbone.eval()

    EMBED_DIM = 512
    model   = DINOv3FaceModel(backbone, embed_dim=EMBED_DIM).to(device)

    TRANSFORM = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset        = FaceDataset(DATA_DIR, TRANSFORM)
    num_identities = len(dataset.identity_map)
    loader         = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    arcface   = ArcFaceLoss(EMBED_DIM, num_identities).to(device)

    # only the projection head + ArcFace class centers are trainable
    optimizer = torch.optim.AdamW([
        {'params': model.projection.parameters()},
        {'params': arcface.parameters()},
    ], lr=1e-3, weight_decay=1e-4)

    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        arcface.train()
        total_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)           # forward: backbone + projection
            loss = arcface(embeddings, labels)   # ArcFace loss
            loss.backward()                      # gradients through projection only
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss / len(loader):.4f}")

    # save only the projection head — backbone is unchanged
    torch.save(model.projection.state_dict(), SAVE_PATH)
    print(f"Saved projection head → {SAVE_PATH}")
```

### Inference (ArcFace is dropped — model only)

```python
def get_embedding(model: DINOv3FaceModel, face_image, device) -> torch.Tensor:
    """
    After training, ArcFace is no longer needed.
    Just run the image through the model → L2 normalize → FAISS-ready vector.
    """
    TRANSFORM = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = TRANSFORM(face_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model(tensor)
    return F.normalize(embedding[0], p=2, dim=0).cpu()
    # → 512-d vector, drop straight into FAISS
```

---

## What Changes in the Pipeline

| | Before | After |
|---|---|---|
| Embedder | DINOv3 raw output | DINOv3 + trained projection head |
| Embedding dim | 768 | 512 |
| FAISS index dim | d=768 | d=512 |
| Pose invariance | Poor | Strong |
| Training data needed | None (zero-shot) | Labeled faces, multiple angles |

---

## Resources to Learn This Properly

- **ArcFace paper**: https://arxiv.org/abs/1801.07698
- **PyTorch transfer learning tutorial**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **InsightFace (pretrained ArcFace models)**: https://github.com/deepinsight/insightface
- **VGGFace2 dataset** (for training data at scale): https://github.com/ox-vgg/vgg_face2
