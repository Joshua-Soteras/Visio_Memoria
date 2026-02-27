import os
import torch
import numpy as np
from torchvision.transforms import v2
import torch.nn.functional as F
from PIL import Image
import resource  # Unix system resource tracking
from dotenv import load_dotenv


'''
model = torch.hub.load(
            REPO_DIR,  #direction where
            'dinov3_vitb16',
            source='local',
            weights=WEIGHTS      
'''



#===================================================
#default model will be dinov3_vitb16 (base model)
cur_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_REPO  = os.path.join(cur_dir, "dinov3")
DEFAULT_MODEL = 'dinov3_vitb16'
#dinov3_vitb16 weights 
load_dotenv() 
WEIGHTS = os.getenv("weights_vitb16")
IMG_DIM = 256


#Get hardware to run device on 
def get_device(): 
    """Pick the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
#===================================================


_instance: "FaceEmbedder | None" = None 
class FaceEmbedder: 

    
    #Dinov3 expect images in a specific format 
    TRANSFORM = v2.Compose([

        #Coverts PIL / NUMPY to Pytorch Tensor 
        #Changes shape fromo (H,W,3) to (3,H,W)
        #Pytorch does channels first 
        v2.ToImage(),

        #Dinov3 works with images that a multiple of 16
        #resizes the image to given dimensions 
        #antialias to smoothen edges while downscaling 
        v2.Resize((IMG_DIM, IMG_DIM), antialias = True),

        #convert pixel to values from 0 -255 integers to 0.0 - 1.0
        #scale = True -> divide by 255 -> values will be 0-1
        #easier for neural nets to work with floats
        v2.ToDtype(torch.float32, scale = True),


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


    def __init__(
            self,
            model_repo:str = MODEL_REPO, 
            model_name = DEFAULT_MODEL, 
            weights =  WEIGHTS,
            device: torch.device = None, 
            ):
        
        """
        This is assuming you are using dinov3 models. 
        Args: 
            model_repo -> where model is stored
            model_name -> name of model
            weights -> weights of model for dinov3
            device -> device to load model 
        """
        
        self.device = device or get_device()


        #Loading Model 
        print("\n---Config: Loading Model and Mode")
        print("Loading ${self.model_name} model and using ${self.device} device")
        try: 
            self.model = torch.hub.load(
                model_repo, 
                model_name, 
                source = 'local',
                weights = weights
            )
            
            self.model = self.model.to(self.device)
            print("Model has succesfully loaded ")
        
        except Exception as e: 
            print("ERROR: Dinov3 Model could not be loaded")
            print("e = ${e}")
            exit()


        # Get embedding dimension from a dummy forward pass
        with torch.inference_mode(): 
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            out = self.model(dummy)
            self.embed_dim = out.shape[-1]

        print(f"Ready — embedding dim: {self.embed_dim}")

    

    #===================================================
    #Cropping face from yolov8 bounding box -> passing to DINOv3 for feature extraction 
    #===================================================
    @staticmethod
    def crop_face_from_frame(frame: np.ndarray, bbox, padding: float = 0.2) -> Image.Image: 

        """
        Crop a face from an OpenCV frame using YOLOv8-face bbox.

        Args:
            frame: OpenCV BGR frame (H, W, 3)
            bbox: [x1, y1, x2, y2] from YOLOv8-face detection
            padding: Extra padding around the face (0.2 = 20%)

        Returns:
            PIL Image of the cropped face (RGB)
        """

        #frame from image captured 
        """
            frame comes a values of [height, width, channels color]
        """ 
        height, width = frame.shape[:2]


        """
        grab the coordinates from the bounding box 
        """
        x1, y1, x2, y2 = [int(coordinate) for coordinate in bbox]

        #adding padding to capture some context (hair, ears, etc.)
        face_width = x2 - x1 
        face_h = y2 - y1 

        pad_x = int(face_width * padding) 
        pad_y = int(face_h * padding)

        """
        Creating padding
        Setting max and min boundaries to "cut
        if we don't we might be cutting out the box
        """
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y) 
        
        
        #Crop and convert BGR → RGB → PIL 
        face_crop = frame[y1:y2 , x1:x2]
        face_rgb = face_crop[:,:,::-1]

        '''
        returns a PIL objects 
        from raw numbers to fully formed image 
        why: any modern ai libraries (like pytorch facenet, or clip) reuqure PIL Images
        '''
        return Image.fromarray(face_rgb)
        
        
        
    #===================================================
    #Cropping face from yolov8 bounding box -> passing to DINOv3 for feature extraction 
    #================================================== 
    def get_embedding(self, face_image: Image.Image) -> torch.Tensor: 

        '''
            Extract an L2-normalized embedding from a face crop 
        '''

        #preproces: resize, to tensor, normalize 
        tensor = self.TRANSFORM(face_image).unsqueeze(0).to(self.device)

        #forward pass 
        with torch.inference_mode(): 
            embedding = self.model(tensor)

        #L2 normalize so cosine similarity = dot product
        #normailzing/formating the output outside of the model to create mathematcial comparisions 
        embedding = F.normalize (embedding[0] , p =2 , dim = 0)

        return embedding.cpu()
    

    def get_embeddings_batch(self, face_images: list[Image.Image]) -> torch.Tensor: 

        """
            Extarct embeddings for multip;e faces at once(faster than one by one)
            Args: face_images : List of PIL Images
        """

        if not face_images:
            return torch.empty(0, self.embed_dim)

        batch = torch.stack([self.TRANSFORM(img) for img in face_images]).to(self.device)

        with torch.inference_mode():
            embeddings = self.model(batch)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()
    


    @staticmethod 
    def compare(emb1: torch.Tensor, emb2: torch.Tensor) -> float: 
        "cosine simialry between two normalized emb"
        #item(extracts the )
        return torch.dot(emb1 ,emb2).item() 
    
    def is_same_person(self, emb1: torch.Tensor, emb2: torch.Tensor) -> bool:
        """Check if two embeddings are likely the same person."""
        return self.compare(emb1, emb2) >= self.SAME_PERSON_THRESHOLD
    
    
    def find_match(self, query_emb: torch.Tensor, database_embs: torch.Tensor) -> tuple[int, float]:
        """
        Find the best match for a face in the database.

        Args:
            query_emb: Single face embedding, shape (embed_dim,)
            database_embs: All stored embeddings, shape (N, embed_dim)

        Returns:
            (best_index, best_similarity)
            Returns (-1, 0.0) if database is empty
        """
        if database_embs.numel() == 0:
            return -1, 0.0

        similarities = database_embs @ query_emb  # matrix-vector product
        best_idx = similarities.argmax().item()
        best_sim = similarities[best_idx].item()
        return best_idx, best_sim


# ─────────────────────────────────────────────
# Singleton accessor — always use this instead of FaceEmbedder() directly.
# The model (~342 MB) is expensive to load; this ensures it only loads once.
# ─────────────────────────────────────────────

def get_embedder(
    model_repo: str = MODEL_REPO,
    model_name: str = DEFAULT_MODEL,
    weights: str = WEIGHTS,
    device: torch.device = None,
) -> FaceEmbedder:
    """
    Return the shared FaceEmbedder instance, creating it on first call.
    Subsequent calls return the already-loaded instance and ignore arguments.

    Usage:
        embedder = get_embedder()           # first call: loads model
        embedder = get_embedder()           # second call: returns same instance
    """
    global _instance
    if _instance is None:
        _instance = FaceEmbedder(model_repo, model_name, weights, device)
    return _instance


# ─────────────────────────────────────────────
# 6. INTEGRATION EXAMPLE — plug into your YOLOv8-face loop
# ─────────────────────────────────────────────

def example_integration():
    """
    PURPOSE:
        This function is a working template that connects every piece of the pipeline.
        It is NOT production code — it is a reference showing how YOLOv8 face detection
        and DINOv3 embedding extraction fit together inside a live webcam loop.

        Think of it as: "here is the skeleton of the real system."
        The commented-out Step 4 (database matching) is the next thing to implement.

    FLOW:
        Webcam frame
            → YOLO detects all faces in the frame → gives back bounding boxes
                → for each bounding box (each face):
                    → crop the face out of the frame (with padding)
                    → run it through DINOv3 → get a 768-d embedding vector
                    → (TODO) compare vector against database to identify the person
            → show the frame in a window
            → repeat until 'q' is pressed
    """

    import cv2

    # ── cv2.VideoCapture / YOLO setup ──────────────────────────────────────────
    # OpenCV and YOLO are imported here (not at the top of the file) because
    # example_integration() is a standalone demo — the rest of FaceEmbedder doesn't
    # need cv2 or YOLO at all. Keeping imports local avoids forcing those dependencies
    # on anything that just wants to use get_embedder().
    from ultralytics import YOLO

    # Load the YOLOv8-face detector. This reads the .pt weights file from disk.
    # detector is now a callable — pass it a frame, get back detection results.

    path =os.path.join(cur_dir, "yolov8-face", "yolov8n-face.pt")
    detector = YOLO(path)

    # get_embedder() returns the singleton FaceEmbedder instance.
    # If called elsewhere first, this returns the already-loaded model (no reload).
    # If this is the first call, it loads DINOv3 now (~342 MB, takes a few seconds).
    embedder = get_embedder()


    # --- DEBUGGING SECTION ---
    print("Attempting to open camera...")
    # cv2.VideoCapture(0) opens the default webcam (index 0).
    # If 0 fails (wrong camera), try 1 or 2.
    # cap is a stateful object — you call cap.read() repeatedly to pull frames.
    cap = cv2.VideoCapture(0) # Try 0, then 1 if this fails

    if not cap.isOpened():
        print("ERROR: Could not open video source.")
        exit()
    else:
        print("Camera initialized successfully.")

    # ── Main inference loop ────────────────────────────────────────────────────
    # This loop runs once per frame (~30x per second at 30fps).
    # Each iteration: grab a frame → detect → embed → (TODO) match → display.
    while True:

        # cap.read() asks the webcam for the next frame.
        # ret (bool): did the capture succeed?
        # frame (np.ndarray): the raw BGR image, shape [H, W, 3]
        ret, frame = cap.read()
        if not ret:
            # Camera disconnected or stream ended — exit cleanly
            break

        # ── Step 1: Face Detection ─────────────────────────────────────────────
        # Pass the full frame into YOLO. It scans the entire image and returns
        # a list of Results objects — one per image in the batch (here, just 1).
        # verbose=False suppresses YOLO's per-frame console output.
        results = detector(frame, verbose=False)
        

        # results is a list — iterate over each Results object.
        # In single-frame inference there is only results[0], but the loop
        # keeps the code compatible with batch inference later.
        for result in results:

            # .boxes is None if YOLO found zero faces in this frame.
            # Skip to the next result rather than crashing.
            if result.boxes is None:
                continue

          

            # Each box is one detected face.
            # If 3 faces are in the frame, result.boxes has 3 rows.
            for box in result.boxes:

                # .xyxy gives bounding box corners as a tensor: [[x1, y1, x2, y2]]
                # [0] selects the first (only) row, .tolist() converts to a Python list.
                bbox = box.xyxy[0].tolist()     # → [x1, y1, x2, y2]

                # .conf is the model's confidence that this is actually a face (0.0–1.0).
                # .item() extracts the Python float from the single-element tensor.
                conf = box.conf[0].item()

                # Skip weak detections — anything below 50% confidence is unreliable.
                # Raising this threshold reduces false positives (hands, objects YOLO
                # mistakes for faces) at the cost of missing low-confidence real faces.
                if conf < 0.5:
                    continue

                # ── Step 2: Crop Face ──────────────────────────────────────────
                # Use the bbox to cut the face out of the raw frame.
                # crop_face_from_frame() also adds 20% padding and converts BGR→RGB→PIL.
                # Result: a PIL Image ready for the transform pipeline.
                face_pil = embedder.crop_face_from_frame(frame, bbox)

                # ── Step 3: Embed Face ─────────────────────────────────────────
                # Run the PIL Image through TRANSFORM → unsqueeze → DINOv3 → F.normalize.
                # Result: a 1D CPU tensor of shape [768] — the face's "fingerprint".
                # Each person has a unique region of 768D space their embeddings cluster in.
                embedding = embedder.get_embedding(face_pil)
                print(f"Embedding shape: {embedding.shape}")  # torch.Size([768])

                # ── Step 4: Match Against Database (NOT YET IMPLEMENTED) ───────
                # This is where identity recognition happens.
                # find_match() does a cosine similarity search across all stored embeddings.
                # High similarity → same person seen before → greet them.
                # Low similarity → new face → save crop + embedding for labeling.
                #
                # match_idx, similarity = embedder.find_match(embedding, db_embeddings)
                # if similarity >= embedder.SAME_PERSON_THRESHOLD:
                #     print(f"Welcome back, {names[match_idx]}!")
                # elif similarity <= embedder.DIFFERENT_PERSON_THRESHOLD:
                #     print("New face detected — saving for labeling...")

        # ── Display ────────────────────────────────────────────────────────────
        # Show the raw frame in a window. Swap in results[0].plot() here to draw
        # YOLO's bounding boxes on screen (useful for debugging detection quality).
        cv2.imshow("Lumemoria", results[0].plot())

        # waitKey(1) pauses 1ms and checks for a keypress.
        # & 0xFF masks to the last 8 bits (needed on some platforms for correct comparison).
        # Pressing 'q' breaks the loop ancled exits cleanly.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()



# ── Entry point ───────────────────────────────────────────────────────────────
# When this file is run directly (`python FaceEmbedder.py`), call the demo.
# When it is imported by another module, this block is skipped — so importing
# get_embedder() or FaceEmbedder never accidentally triggers the webcam loop.
if __name__ == "__main__":
    example_integration()









