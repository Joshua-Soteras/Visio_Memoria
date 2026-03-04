# Fine-Tuning DINOv3 with ArcFace — Full Learning Guide

A self-contained guide to understanding, collecting data for, training, and integrating
an ArcFace projection head on top of DINOv3 for pose-invariant face recognition.

---

## The Core Idea in One Sentence

> Fine-tuning does NOT change DINOv3. It adds a small trainable layer on top that
> remaps DINOv3's visual features into a space where the same person always lands
> close together — regardless of angle.

---

## What Changes vs What Stays the Same

```
BEFORE fine-tuning:
  face image ──→ DINOv3 ──→ 768-d vector
                              (organized by visual appearance)

AFTER fine-tuning:
  face image ──→ DINOv3 ──→ Projection Head ──→ 512-d vector
                 [frozen]    [trained with         (organized by identity)
                              ArcFace loss]
```

| Component        | Changes? | Notes                                    |
|------------------|----------|------------------------------------------|
| DINOv3 backbone  | No       | Frozen — weights never update            |
| Projection head  | Yes      | This is what training teaches            |
| ArcFace loss     | Yes      | Training signal — discarded at inference |
| FAISS index      | No       | Just change dimension: 768 → 512         |
| SQLite           | No       | Unchanged                                |

---

## Why You Need Training Data

The projection head starts as random weights — it does nothing useful.
Training teaches it by showing the same person from many angles:

```
Training signal:

  person_1/front.jpg  ──→ embedding A  ┐
  person_1/left.jpg   ──→ embedding B  ├── ArcFace says: A, B, C must be CLOSE
  person_1/right.jpg  ──→ embedding C  ┘

  person_2/front.jpg  ──→ embedding D  ── ArcFace says: D must be FAR from A, B, C
```

The more identities and the more angles per identity, the better the model
generalizes to faces it has never seen before.

---

## Step-by-Step Overview

```
Step 1: Collect training data
    → Run collect_faces.py (webcam + YOLOv8)
    → Move head in all directions while pressing 's' to save crops
    → Repeat for every person you want the model to learn from

Step 2: Train the projection head
    → Run arcface_train.py
    → DINOv3 loads frozen
    → Only the projection head + ArcFace class centers train
    → Loss should decrease each epoch
    → arcface_head.pth saved when done

Step 3: Use it in the pipeline
    → Load DINOv3 + arcface_head.pth together
    → Run new face through both
    → Get 512-d embedding
    → Store in / search FAISS as normal
```

---

## Step 1 — Data Collection Script

**How to use:**
- Run the script
- Press `n` → type a person's name → press Enter
- Move your head: look left, right, up, down, tilt — press `s` each time to save
- Aim for 20–30 images per person across different angles
- Press `n` again for a new person
- Press `q` when done

**Where data is saved:**
```
src/visio_memoria/data/faces/
    your_name/
        img_0001.jpg
        img_0002.jpg
        ...
    person_2/
        img_0001.jpg
        ...
```

```python
"""
collect_faces.py

Interactive webcam script for collecting face training data.
Uses YOLOv8-Face to detect and crop faces, saves them organized
by identity name for ArcFace fine-tuning.

Controls:
    n → enter new person name
    s → save current face crop
    q → quit
"""

import os
import cv2
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
# Always resolve paths from this file's location so the script works
# regardless of where it's called from.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                          # src/visio_memoria/
MODEL_PATH  = os.path.join(PROJECT_DIR, "models", "yolov8-face", "yolov8n-face.pt")
DATA_DIR    = os.path.join(PROJECT_DIR, "data", "faces")           # output directory


def get_save_path(identity: str, count: int) -> str:
    """Build and ensure the save path for a face crop."""
    person_dir = os.path.join(DATA_DIR, identity)
    os.makedirs(person_dir, exist_ok=True)       # create folder if it doesn't exist
    return os.path.join(person_dir, f"img_{count:04d}.jpg")


def main():
    # Load YOLOv8-Face — same pattern as yolov8_face_test.py
    model = YOLO(MODEL_PATH)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    current_identity = "unknown"   # current person being captured
    save_count = 0                 # running count of saved images

    print("Controls: [n] new person  [s] save crop  [q] quit")
    print(f"Current identity: {current_identity}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 face detection
        # conf=0.4 means only keep detections the model is 40%+ confident about
        results = model(frame, conf=0.4, verbose=False)

        face_crop = None   # will hold the cropped face if one is detected

        # Iterate over detected faces in this frame
        for box in results[0].boxes:
            # .xyxy gives [x1, y1, x2, y2] — the bounding box corners
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Draw bounding box on the live frame so user can see detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop the face from the frame
            # frame[y1:y2, x1:x2] slices the numpy array to the bounding box region
            face_crop = frame[y1:y2, x1:x2]
            break   # only use the first detected face per frame

        # Overlay instructions and current status on the video feed
        cv2.putText(frame, f"Identity: {current_identity}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Saved: {save_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "[n] new person  [s] save  [q] quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("n"):
            # Enter new identity name via terminal
            cv2.destroyAllWindows()
            current_identity = input("Enter person name: ").strip()
            save_count = 0
            print(f"Now collecting for: {current_identity}")
            print("Controls: [n] new person  [s] save crop  [q] quit")

        elif key == ord("s"):
            # Save the current face crop to disk
            if face_crop is not None and face_crop.size > 0:
                path = get_save_path(current_identity, save_count)
                cv2.imwrite(path, face_crop)
                save_count += 1
                print(f"Saved: {path}")
            else:
                print("No face detected — move into frame first.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Data saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
```

---

## Step 2 — Training Script

### What the training loop does, step by step

```
Each epoch:
    for each batch of (images, identity_labels):

        1. images → DINOv3 → 768-d features     [no gradients, backbone is frozen]
        2. features → projection head → 512-d embeddings  [gradients flow here]
        3. ArcFace computes loss:
              a. L2-normalize embeddings + class weight vectors
              b. cosine similarity between each embedding and all class centers
              c. convert cosine → angle
              d. add margin to the correct class angle
              e. convert back to cosine → scale → cross entropy
        4. loss.backward()   → gradients flow back through projection + ArcFace
                             → DINOv3 gets NO gradients (frozen)
        5. optimizer.step()  → only projection head + class centers update

After all epochs:
    save projection head weights → arcface_head.pth
```

### Gradient flow diagram

```
ArcFace loss
     ↑  gradients flow up
Projection head   ← weights update ✓
     ↑  gradients flow up
DINOv3 backbone   ← torch.no_grad() blocks gradients ✗
     ↑  no gradients
Input image
```

### Full training script

```python
"""
arcface_train.py

Fine-tunes DINOv3 ViT-B/16 with an ArcFace projection head for
pose-invariant face recognition.

Pipeline:
    face images (folder per identity)
        → FaceDataset
        → DINOv3 (frozen backbone)
        → projection head (trainable Linear + BatchNorm)
        → ArcFace loss (trainable class centers)
        → saved arcface_head.pth

Usage:
    python arcface_train.py
    (run from anywhere — paths resolve from this file's location)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from dotenv import load_dotenv
from PIL import Image


# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR    = os.path.join(PROJECT_DIR, "models", "dinov3")
DATA_DIR    = os.path.join(PROJECT_DIR, "data", "faces")
SAVE_PATH   = os.path.join(PROJECT_DIR, "models", "arcface_head.pth")

load_dotenv()
WEIGHTS = os.getenv("weights_vitb16")


# ── Device ───────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """MPS (Apple Silicon) → CUDA (NVIDIA) → CPU fallback."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    """
    Loads face images from a folder-per-identity structure.

    Expected layout:
        data/faces/
            alex/
                img_0001.jpg
                img_0002.jpg
            josh/
                img_0001.jpg

    Each subfolder name becomes an identity label.
    Labels are assigned as integers automatically (sorted alphabetically).

    Why sort alphabetically?
        Sorting ensures the identity → integer mapping is deterministic.
        Running the script twice gives the same label assignments.
    """

    def __init__(self, root_dir: str, transform):
        self.transform    = transform
        self.samples      = []   # list of (image_path, integer_label)
        self.identity_map = {}   # name → integer (e.g. "alex" → 0, "josh" → 1)

        for label_idx, identity in enumerate(sorted(os.listdir(root_dir))):
            identity_dir = os.path.join(root_dir, identity)

            # Skip files — only process directories
            if not os.path.isdir(identity_dir):
                continue

            self.identity_map[identity] = label_idx

            for img_file in sorted(os.listdir(identity_dir)):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(identity_dir, img_file), label_idx)
                    )

        print(f"Dataset: {len(self.identity_map)} identities, {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")   # always load as RGB
        return self.transform(image), label


# ── ArcFace Loss ─────────────────────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).
    Paper: https://arxiv.org/abs/1801.07698

    The key idea:
        Standard softmax classifies by cosine similarity.
        ArcFace adds a fixed angular margin (m) to the CORRECT class angle
        before computing softmax. This forces embeddings to cluster tightly
        and stay angularly far apart from other identities.

    Parameters:
        embed_dim      → size of the embedding vector (512)
        num_identities → number of people in your dataset
        scale          → amplifies cosine logits (usually 64)
                         without this, gradients would be too small to learn
        margin         → angular penalty in radians (usually 0.5 ≈ 28 degrees)
                         larger margin = tighter clusters = harder to train
    """

    def __init__(self, embed_dim: int, num_identities: int, scale=64.0, margin=0.5):
        super().__init__()
        self.scale  = scale
        self.margin = margin

        # Class weight matrix: one learnable vector per identity
        # These are the "class centers" — where each identity should cluster
        # Shape: (num_identities, embed_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_identities, embed_dim))

        # Xavier uniform init: prevents weights from starting too large or small
        # Proper initialization is critical for stable training
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Step 1: L2-normalize both embeddings and class weight vectors
        # After normalization, dot product equals cosine similarity
        # This puts everything on a unit hypersphere — only angles matter
        embeddings = F.normalize(embeddings, p=2, dim=1)   # (B, embed_dim)
        weight     = F.normalize(self.weight,    p=2, dim=1)   # (num_ids, embed_dim)

        # Step 2: cosine similarity between each embedding and every class center
        # F.linear computes: output[i][j] = embeddings[i] · weight[j]
        # Result shape: (B, num_identities)
        cosine = F.linear(embeddings, weight)

        # Clamp to valid range for acos — floating point can slightly exceed [-1, 1]
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # Step 3: convert cosine similarity → angle (theta)
        # acos maps cosine values back to angles in [0, π]
        theta = torch.acos(cosine)   # (B, num_identities)

        # Step 4: add margin ONLY to the correct class angle
        # one_hot[i][labels[i]] = 1.0, all other entries = 0.0
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # theta + margin for correct class, theta unchanged for all others
        # The margin makes the model work harder to get the correct class right
        output = torch.cos(theta + self.margin * one_hot)

        # Step 5: scale up and compute cross-entropy
        # Without scaling, the cosine values (−1 to 1) produce near-zero gradients
        output *= self.scale   # (B, num_identities)

        return F.cross_entropy(output, labels)


# ── Model ────────────────────────────────────────────────────────────────────
class DINOv3FaceModel(nn.Module):
    """
    Frozen DINOv3 backbone + small trainable projection head.

    Why freeze the backbone?
        DINOv3 was pretrained on 1.4 billion images using self-supervised learning.
        Its features are already extremely powerful. Training the full model requires:
          - Far more labeled face data
          - Much more GPU compute (hours → days)
          - Risk of destroying the pretrained features (catastrophic forgetting)

        Freezing it and only training the projection head is called "linear probing"
        or "head fine-tuning" — the most data-efficient form of transfer learning.

    Why BatchNorm in the projection head?
        BatchNorm normalizes each feature dimension across the batch.
        This keeps embedding values in a consistent range during training,
        which stabilizes the ArcFace loss computation and speeds up convergence.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int = 512):
        super().__init__()

        # Freeze all DINOv3 parameters — no gradients will flow through them
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Projection head: remaps 768-d DINOv3 features → 512-d identity embeddings
        # This is the ONLY part that trains
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),     # 768 = DINOv3 ViT-B/16 output dimension
            nn.BatchNorm1d(embed_dim),     # stabilize training
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract DINOv3 features — torch.no_grad() enforces no gradient tracking
        # Even though backbone params have requires_grad=False, this is explicit
        with torch.no_grad():
            features = self.backbone(x)       # (B, 768)

        # Project to identity-organized embedding space
        embeddings = self.projection(features)   # (B, 512)
        return embeddings


# ── Training ─────────────────────────────────────────────────────────────────
def train():
    device = get_device()
    print(f"Device: {device}")

    # Load DINOv3 backbone (same pattern as FaceEmbedder.py)
    print("Loading DINOv3...")
    backbone = torch.hub.load(
        REPO_DIR,
        "dinov3_vitb16",
        source="local",
        weights=WEIGHTS,
    )
    backbone.eval()   # set to eval mode: disables dropout, fixes BatchNorm stats

    # Build model (backbone frozen, projection head trainable)
    EMBED_DIM = 512
    model = DINOv3FaceModel(backbone, embed_dim=EMBED_DIM).to(device)

    # Image preprocessing — must match DINOv3's training distribution
    # Wrong normalization = garbage embeddings (critical!)
    TRANSFORM = v2.Compose([
        v2.ToImage(),                              # PIL → tensor, shape (3, H, W)
        v2.Resize((256, 256), antialias=True),     # DINOv3 expects multiples of 16
        v2.ToDtype(torch.float32, scale=True),     # 0–255 uint8 → 0.0–1.0 float32
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),            # ImageNet channel means
            std=(0.229, 0.224, 0.225),             # ImageNet channel std devs
        ),
    ])

    # Load dataset
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: No data found at {DATA_DIR}")
        print("Run collect_faces.py first.")
        return

    dataset        = FaceDataset(DATA_DIR, TRANSFORM)
    num_identities = len(dataset.identity_map)

    if num_identities < 2:
        print("ERROR: Need at least 2 identities to train.")
        return

    # DataLoader: feeds batches of images to the model during training
    # shuffle=True randomizes the order each epoch — critical for generalization
    # num_workers=0 on MPS (macOS multiprocessing can cause issues)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    # ArcFace loss has its own trainable class weight matrix
    arcface = ArcFaceLoss(EMBED_DIM, num_identities).to(device)

    # Optimizer: only train projection head + ArcFace class centers
    # DINOv3 backbone params are frozen — no point including them
    # AdamW: Adam with weight decay, better generalization than plain Adam
    optimizer = torch.optim.AdamW(
        [
            {"params": model.projection.parameters()},
            {"params": arcface.parameters()},
        ],
        lr=1e-3,        # learning rate: how big each gradient step is
        weight_decay=1e-4,  # L2 regularization: penalizes large weights
    )

    # Training loop
    EPOCHS = 30
    print(f"\nTraining for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()     # projection head: enable training mode (BatchNorm uses batch stats)
        arcface.train()
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Zero gradients from the previous step
            # If you don't do this, gradients accumulate and training explodes
            optimizer.zero_grad()

            # Forward pass: image → DINOv3 (frozen) → projection head → 512-d embedding
            embeddings = model(images)

            # Compute ArcFace loss
            loss = arcface(embeddings, labels)

            # Backward pass: compute gradients for projection head + ArcFace weights
            # DINOv3 backbone is frozen so its gradients are not computed
            loss.backward()

            # Update projection head + ArcFace class centers using computed gradients
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1:3d}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # Save ONLY the projection head weights
    # DINOv3 backbone is unchanged so there's no need to save it
    torch.save(model.projection.state_dict(), SAVE_PATH)
    print(f"\nProjection head saved → {SAVE_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    train()
```

---

## Step 3 — Using the Trained Model at Inference

After training, ArcFace loss is discarded. You only need the model.

```python
"""
DINOv3FaceModel.py — inference wrapper

Loads DINOv3 + trained ArcFace projection head.
Returns 512-d L2-normalized embeddings ready for FAISS.

Why 512-d instead of 768-d?
    The projection head maps 768 → 512.
    These 512-d vectors are identity-organized (trained with ArcFace),
    whereas raw 768-d DINOv3 vectors are visually organized.
    FAISS index must be rebuilt with d=512 to match.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from dotenv import load_dotenv
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.join(SCRIPT_DIR, "dinov3")
HEAD_PATH  = os.path.join(SCRIPT_DIR, "arcface_head.pth")

load_dotenv()
WEIGHTS = os.getenv("weights_vitb16")

TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class DINOv3FaceModel(nn.Module):
    """Frozen DINOv3 + loaded projection head. Ready for inference only."""

    def __init__(self, backbone: nn.Module, embed_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.projection(features)


def load_model(device: torch.device) -> DINOv3FaceModel:
    """Load DINOv3 backbone + trained projection head."""
    backbone = torch.hub.load(REPO_DIR, "dinov3_vitb16", source="local", weights=WEIGHTS)
    backbone.eval()

    model = DINOv3FaceModel(backbone).to(device)

    # Load the trained projection head weights
    state_dict = torch.load(HEAD_PATH, map_location=device)
    model.projection.load_state_dict(state_dict)
    model.eval()   # projection head: disable BatchNorm training mode

    return model


def get_embedding(model: DINOv3FaceModel, face_image: Image.Image, device) -> torch.Tensor:
    """
    Run a face crop through the fine-tuned model.

    Returns a 512-d L2-normalized embedding, ready to insert into FAISS.
    L2 normalization means cosine similarity = dot product, which is what
    FAISS IndexFlatIP computes.
    """
    tensor = TRANSFORM(face_image).unsqueeze(0).to(device)   # (1, 3, 256, 256)

    with torch.inference_mode():   # faster than no_grad — disables version tracking
        embedding = model(tensor)  # (1, 512)

    # L2 normalize: scale vector to unit length so only angle matters
    embedding = F.normalize(embedding[0], p=2, dim=0)   # (512,)
    return embedding.cpu()   # move to CPU for storage/FAISS compatibility


# Quick test
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_model(device)

    # Test with a dummy image
    dummy = Image.new("RGB", (256, 256))
    emb = get_embedding(model, dummy, device)
    print(f"Embedding shape: {emb.shape}")   # should print: torch.Size([512])
    print(f"Embedding norm:  {emb.norm():.4f}")   # should be close to 1.0
```

---

## What Changes in FAISS After Fine-Tuning

```python
# Before fine-tuning:
index = faiss.IndexFlatIP(768)   # d=768 for raw DINOv3

# After fine-tuning:
index = faiss.IndexFlatIP(512)   # d=512 for projection head output
```

Everything else in the FAISS + SQLite pipeline stays identical.
The only change is the dimension of the vectors being stored and queried.

---

## How to Know Training is Working

Watch the loss decrease across epochs:

```
Epoch   1/30 — Loss: 12.3421   ← high at start (random weights)
Epoch   5/30 — Loss:  8.2103
Epoch  10/30 — Loss:  5.4871
Epoch  20/30 — Loss:  2.9034
Epoch  30/30 — Loss:  1.2451   ← lower = embeddings are tighter
```

If loss is not decreasing:
- Learning rate may be too high → try lr=1e-4
- Not enough data → collect more images per person
- Only one identity → need at least 2

---

## Summary: What You Built

```
Your custom model = DINOv3 (Meta, 86M params) + your trained projection head

The projection head learned:
    - From YOUR face data
    - Collected from YOUR webcam
    - Organized by YOUR labels

Result: a model that produces 512-d embeddings where angle variation
        collapses into tight clusters per identity — your identity.
```

---

## References
- ArcFace paper: https://arxiv.org/abs/1801.07698
- DINOv3 repo: https://github.com/facebookresearch/dinov2
- PyTorch transfer learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- InsightFace (pretrained ArcFace alternative): https://github.com/deepinsight/insightface
