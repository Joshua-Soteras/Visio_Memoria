"""
pipeline.py — Lumemoria Full Pipeline (v2)
All modules wired together:
  YOLOv8-face → DINOv2 → FAISS → Greeter (TTS) → Activity Tracker

Run:     python pipeline.py
Quit:    press 'q'
Label:   press 'l' (CLI) or 'g' (Gradio UI in browser)
Stats:   press 's'

Requires:
  pip install -r requirements.txt
  pip install faiss-cpu gradio pyttsx3  # optional extras
"""

# cv2 is OpenCV — a computer vision library written in C++ with Python bindings.
# It handles everything visual in this project:
#   - Opening a camera and reading frames (VideoCapture, cap.read)
#   - Drawing rectangles, text, and overlays on images (rectangle, putText)
#   - Displaying image windows (imshow) and capturing keyboard input (waitKey)
#   - Reading image files from disk (imread)
import cv2

# time is a standard Python library for time-related functions.
# We use time.time(), which returns the current time in seconds since the Unix epoch
# (Jan 1, 1970). Subtracting two time.time() calls gives elapsed seconds.
# This powers both the FPS counter and the greeting/save cooldowns.
import time

# threading is a standard Python library for running code concurrently.
# Python can only run one thread at a time (due to the Global Interpreter Lock),
# but threads are still useful for I/O-bound work — like running a web server (Gradio)
# while the camera loop continues in the main thread simultaneously.
import threading

# datetime is a standard Python library for working with dates and times as objects.
# datetime.now() returns a datetime object representing the current moment.
# Unlike time.time() (which gives a raw float of seconds), datetime gives you structured
# access to year, month, day, hour, minute — and lets you format it as a readable string.
from datetime import datetime

# FaceEmbedder — our module that:
#   1. Crops a face out of a raw camera frame using a bounding box
#   2. Runs DINOv2 (a vision transformer) to produce a 768-number fingerprint (embedding)
#      that encodes the face's identity as a vector in mathematical space.
from face_embedder import FaceEmbedder

# FaceDatabaseFAISS — our module that:
#   - Stores face embeddings persistently (on disk, via SQLite + .pt files)
#   - Uses FAISS (Facebook AI Similarity Search) to search billions of vectors fast
#   - Tracks known persons, sightings, and an unlabeled queue for human review
# We alias it as FaceDatabase to keep the rest of the code readable.
from face_database_faiss import FaceDatabaseFAISS as FaceDatabase

# Greeter — our module that manages the text-to-speech (TTS) greeting system.
# GreetingEvent is a dataclass (a simple data container object) that bundles together
# all the info a greeting needs: who the person is, how many times they've visited, etc.
# from greeter import Greeter, GreetingEvent

# ActivityTracker — our module that:
#   - Logs notes and visit context (time of day, day of week) per person
#   - Generates summaries and pattern analysis (e.g. "usually visits on Monday mornings")
from activity_tracker import ActivityTracker


class Lumemoria:
    """
    Full recognition pipeline.

    Frame-by-frame flow:
      1. YOLOv8-face → bounding boxes + landmarks
      2. Crop faces → PIL images
      3. DINOv2 → 768-dim embeddings
      4. FAISS → match against database
      5a. Known   → greet (TTS) + log sighting + track activity
      5b. Unknown → queue for labeling
    """

    def __init__(
        self,
        # Path to the YOLOv8-face weights file (.pt = PyTorch weights format).
        # "yolov8n-face.pt" — the 'n' stands for 'nano', the smallest/fastest model variant.
        yolo_model_path: str = "yolov8n-face.pt",
        # Which DINOv2 variant to use. "dinov2_vitb14" = Vision Transformer Base, patch size 14.
        # Larger model → better accuracy, slower inference.
        dinov3_model: str = "dinov2_vitb14",
        # Where to store all persistent data: SQLite DB, face images, embeddings, FAISS index.
        db_path: str = "./lumemoria_data",
        # Minimum YOLO confidence to treat a detection as a real face.
        # 0.5 = must be at least 50% confident it's a face.
        # Lower → more detections but more false positives. Higher → fewer but more reliable.
        detection_conf: float = 0.5,
        # FAISS similarity threshold for declaring a match. Range is 0.0–1.0 (cosine similarity).
        # 0.65 = embeddings must be at least 65% similar (in angular terms) to count as the same person.
        match_threshold: float = 0.65,
        # Seconds to wait before re-greeting the same person.
        # 300 = 5 minutes. Prevents greeting the same person every single frame they're in view.
        greet_cooldown: int = 300,
        # Seconds to wait before saving another unknown face to the queue.
        # 30 = saves at most one unknown face every 30 seconds to avoid filling disk storage.
        unknown_cooldown: int = 30,
        # Which text-to-speech engine to use. "auto" = let the Greeter pick the best available.
        tts_backend: str = "auto",
    ):
        # YOLO is imported here (inside __init__) instead of at the top of the file.
        # This is called a lazy import — ultralytics is only loaded when Lumemoria() is created.
        # Reason: ultralytics is a heavy package, slow to import. If something else in the
        # file fails first, we don't waste time loading YOLO unnecessarily.
        from ultralytics import YOLO

        print("=" * 50)
        print("  LUMEMORIA v2 — Initializing")
        print("=" * 50)

        # ── 1. Detection ──
        print("\n[1/5] Loading YOLOv8-face...")
        # YOLO() loads the model weights from disk into memory and prepares the model
        # for inference. After this line, self.detector is a callable object —
        # you call it like a function: self.detector(frame) → returns detections.
        self.detector = YOLO(yolo_model_path)
        # Store the threshold so _extract_detections() can filter by it later.
        self.detection_conf = detection_conf

        # ── 2. Embedding ──
        print("[2/5] Loading DINOv2...")
        # FaceEmbedder loads DINOv2 into memory and sets up the image transform pipeline.
        # Loading takes a few seconds (342MB model). After this, self.embedder is ready
        # to receive PIL images and return 768-dimensional embedding tensors.
        self.embedder = FaceEmbedder(model_name=dinov3_model)

        # ── 3. Database (FAISS) ──
        print("[3/5] Loading face database (FAISS)...")
        # FaceDatabase sets up the SQLite database and FAISS index at the given path.
        # If lumemoria_data/ doesn't exist, it creates it. If it does, it loads existing data.
        self.db = FaceDatabase(db_path)

        # Attempt to load a previously saved FAISS index from disk (faiss_index.bin).
        # Loading a saved index is faster than rebuilding it from all stored embeddings.
        # If the file doesn't exist (first run), load_faiss_index() returns False —
        # FaceDatabase already rebuilt the index from SQLite embeddings during __init__.
        if not self.db.load_faiss_index():
            print("  No saved index, rebuilt from embeddings")

        # ── 4. Greeter (TTS) ──
        # print("[4/5] Initializing greeter...")
        # self.greeter = Greeter(backend=tts_backend)
        # self.greeter.start()

        # ── 5. Activity Tracker ──
        print("[5/5] Initializing activity tracker...")
        # ActivityTracker needs direct access to the SQLite connection to read/write
        # its own tables (activity_log, etc.). We pass self.db.conn — the raw SQLite
        # connection object — rather than the whole FaceDatabase, to keep it decoupled.
        self.activity = ActivityTracker(self.db.conn)

        # ── Config ──
        # Store the thresholds as instance variables so all methods can access them
        # without having to re-read the original parameters.
        self.match_threshold = match_threshold
        self.greet_cooldown = greet_cooldown
        self.unknown_cooldown = unknown_cooldown

        # ── Cooldown trackers ──
        # dict[int, float]: maps person_id → timestamp of last greeting.
        # Example: {7: 1714000200.0, 12: 1714001500.0}
        # dict.get(person_id, 0) returns 0 if this person has never been greeted —
        # 0 is the Unix epoch (year 1970), so "now - 0" is always > cooldown. First greet always fires.
        self._last_greet: dict[int, float] = {}

        # float: timestamp of the last time we saved an unknown face.
        # Single value (not a dict) because we don't track per-identity — unknowns have no ID yet.
        self._last_unknown_save: float = 0

        # ── Gradio thread handle ──
        # Stores a reference to the Gradio web UI thread (or None if not started).
        # threading.Thread | None is a type hint — the pipe (|) means "either type is valid here".
        # We need this reference to check later whether Gradio is already running
        # (via thread.is_alive()) before starting a second instance.
        self._gradio_thread: threading.Thread | None = None

        stats = self.db.stats()
        print(f"\n  Known: {stats['known_persons']} persons, "
              f"{stats['total_embeddings']} embeddings, "
              f"FAISS: {stats.get('faiss_index_type', 'N/A')}")
        # stats.get('faiss_index_type', 'N/A') — .get() with a default is safer than stats['key']
        # because if 'faiss_index_type' is missing from the dict, it returns 'N/A' instead of crashing.
        print(f"  Unlabeled queue: {stats['unlabeled_queue']}")
        print("\n✅ Lumemoria ready!\n")



    # ─────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────
    def run(self, camera_index: int = 0):
        """Main camera loop."""
        # cv2.VideoCapture opens the camera at the given index.
        # Index 0 = the first camera the OS finds (built-in webcam on most laptops).
        # Index 1, 2... = additional cameras (e.g. a USB webcam added later).
        # Under the hood, OpenCV talks to the OS camera driver and sets up a frame buffer
        # that continuously fills with incoming frames from the hardware.
        cap = cv2.VideoCapture(camera_index)

        # cap.isOpened() checks whether the camera was successfully opened.
        # It can fail if: the camera doesn't exist, another app has exclusive access,
        # or the driver crashed. Always check before entering the loop.
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return

        print("Controls:")
        print("  q → quit")
        print("  l → label faces (CLI)")
        print("  g → label faces (Gradio browser UI)")
        print("  s → stats")
        print("  a → add activity note")
        print("  p → person details")
        print()

        # FPS (frames per second) tracking. We count frames over 1-second windows.
        fps_counter = 0           # frames counted in the current 1-second window
        fps_time = time.time()    # timestamp when the current 1-second window started
        fps_display = 0           # FPS value shown on screen (updated once per second)

        # try/finally guarantees the cleanup block runs no matter how the loop ends —
        # whether the user pressed 'q', an exception was raised, or the camera disconnected.
        # Without this, a crash would leave the camera open and windows frozen on screen.
        try:
            while True:
                # cap.read() grabs one frame from the camera buffer.
                # Returns two values:
                #   ret   — bool: True if a frame was successfully retrieved,
                #                 False if the camera failed or was disconnected.
                #   frame — numpy array with shape (height, width, 3), dtype uint8.
                #           Color order is BGR (Blue-Green-Red) — OpenCV's default.
                #           BGR is the reversed order of the more common RGB. Historical quirk of OpenCV.
                ret, frame = cap.read()

                # If ret is False, the camera stream has ended. Exit the loop cleanly.
                if not ret:
                    break

                # ── Detect ──
                # self.detector is the YOLO model object. Calling it like a function runs inference.
                # verbose=False suppresses YOLO's per-frame terminal output (speed stats, counts).
                # results is a list of Results objects — one per image passed in.
                # We always pass 1 frame, so results always has exactly 1 element.
                results = self.detector(frame, verbose=False)

                # _extract_detections() unpacks the YOLO Results into two parallel lists:
                #   detections     — list of (bbox, conf) tuples, one per face found
                #   landmarks_list — list of keypoint arrays (or None), same length as detections
                # "Parallel" means index 0 of detections corresponds to index 0 of landmarks_list.
                detections, landmarks_list = self._extract_detections(results)

                # ── Process each face ──
                # enumerate() gives both the index (i) and the value (bbox, conf).
                # We need i to look up the matching landmarks at the same position in landmarks_list.
                for i, (bbox, conf) in enumerate(detections):
                    # Safely grab landmarks for this face.
                    # The "i < len" guard is a safety net in case the two lists somehow differ in length.
                    landmarks = landmarks_list[i] if i < len(landmarks_list) else None
                    self._process_face(frame, bbox, conf, landmarks)

                # ── Draw ──
                # After all faces are processed, draw boxes + overlays onto the frame.
                # Important: OpenCV drawing operations modify the frame array in-place (no copy).
                # We draw AFTER processing so drawing overhead doesn't slow down recognition.
                self._draw_frame(frame, detections, fps_display)

                # ── FPS ──
                fps_counter += 1
                # time.time() - fps_time = seconds elapsed since the last reset.
                # Once a full second has passed, snapshot the frame count as the display value.
                if time.time() - fps_time >= 1.0:
                    fps_display = fps_counter  # e.g. 28 frames → display "FPS: 28"
                    fps_counter = 0
                    fps_time = time.time()     # start the next 1-second window now

                # ── Display ──
                # cv2.imshow() renders the frame in an OS window titled "Lumemoria".
                # The window is created automatically on the first call; subsequent calls update it.
                # The frame already has bounding boxes + text drawn on it by _draw_frame above.
                cv2.imshow("Visio Memoria", frame)

                # cv2.waitKey(1) does two things at once:
                #   1. Gives OpenCV 1 millisecond to process OS events (resize, repaint the window).
                #      Without this, the window freezes and shows a grey/black screen.
                #   2. Returns the ASCII code of any key pressed during that 1ms window.
                #      Returns -1 if no key was pressed.
                # & 0xFF masks the result to the lowest 8 bits. On some Linux/Windows systems,
                # waitKey returns extra high bits set — the mask strips them for safe comparison.
                key = cv2.waitKey(1) & 0xFF

                # ord("q") converts the character "q" to its ASCII integer value (113).
                # We compare integers (not strings) because waitKey returns an int.
                if key == ord("q"):
                    break
                elif key == ord("l"):
                    self._label_cli()
                elif key == ord("g"):
                    self._launch_gradio()
                elif key == ord("s"):
                    self._print_stats()
                elif key == ord("a"):
                    self._add_activity_note()
                elif key == ord("p"):
                    self._show_person_details()

        finally:

            # This block runs whether the loop ended normally ('q') or crashed with an exception.
            # It ensures we never leave hardware in a dangling state.
            # self.greeter.stop()            # signal the TTS thread to shut down gracefully
            self.db.save_faiss_index()     # flush the FAISS index to disk before exiting
            cap.release()                  # release the camera back to the OS (other apps can use it)
            cv2.destroyAllWindows()        # close every OpenCV window (without this, windows linger)
            print("\nLumemoria stopped. FAISS index saved.")



    # ─────────────────────────────────────────
    # DETECTION
    # ─────────────────────────────────────────
    def _extract_detections(self, results) -> tuple[list, list]:
        """Extract bboxes, confidences, and landmarks from YOLO results."""
        detections = []
        landmarks_list = []

        # results is a list of Results objects — one per image passed to self.detector().
        # Since we always call it with a single frame, this loop runs exactly once.
        # The for-loop structure is YOLO's standard output format for any batch size.
        for result in results:
            # result.boxes is a Boxes object containing all face detections in this image.
            # It can be None if YOLO found nothing (e.g. a completely empty or dark frame).
            if result.boxes is None:
                continue

            # Iterate over each detected face in this result.
            # enumerate() gives us i (the index) alongside box (the individual detection).
            # We need i to fetch the matching landmarks from result.keypoints.xy[i].
            for i, box in enumerate(result.boxes):
                # box.conf is a tensor holding the confidence score for this detection.
                # [0] unwraps it from a 1-element tensor to a scalar tensor.
                # .item() converts that scalar tensor to a plain Python float (e.g. 0.873).
                conf = box.conf[0].item()

                # Filter out weak detections. Only process faces above our confidence threshold.
                if conf >= self.detection_conf:
                    # box.xyxy[0] gives the bounding box in [x1, y1, x2, y2] format:
                    #   x1, y1 = top-left corner pixel coordinates
                    #   x2, y2 = bottom-right corner pixel coordinates
                    # [0] unwraps from a 2D tensor row to a 1D tensor of 4 values.
                    # .tolist() converts tensor → plain Python list [float, float, float, float].
                    bbox = box.xyxy[0].tolist()
                    detections.append((bbox, conf))

                    # Landmarks are the 5 key facial points YOLOv8-face can detect:
                    # left eye, right eye, nose tip, left mouth corner, right mouth corner.
                    # They're optional — not all models produce them.
                    # hasattr() checks at runtime whether 'keypoints' exists on this result object.
                    if hasattr(result, "keypoints") and result.keypoints is not None:
                        try:
                            # result.keypoints.xy[i] — select landmarks for the i-th face.
                            # .xy gives a tensor of shape (num_faces, 5, 2): 5 points × (x, y).
                            # [i] selects this face's 5 points → shape (5, 2).
                            # .cpu() moves the tensor from GPU to CPU memory (required before .numpy()).
                            # .numpy() converts PyTorch tensor → NumPy array (standard format downstream).
                            kps = result.keypoints.xy[i].cpu().numpy()
                            landmarks_list.append(kps)
                        except (IndexError, AttributeError):
                            # If anything goes wrong (e.g. index out of range for this result),
                            # append None to keep landmarks_list the same length as detections.
                            landmarks_list.append(None)
                    else:
                        landmarks_list.append(None)

        return detections, landmarks_list

    # ─────────────────────────────────────────
    # CORE: PROCESS A SINGLE FACE
    # ─────────────────────────────────────────

    def _process_face(self, frame, bbox, conf, landmarks=None):
        """Crop → embed → match → greet or queue."""
        # Snapshot the current time once at the top. Reusing `now` avoids tiny timing
        # drift if time.time() were called multiple times throughout this function.
        now = time.time()

        # ── Step 1: Crop ──
        # crop_face_from_frame() takes the raw BGR numpy frame and the [x1,y1,x2,y2] bbox.
        # It crops that region, adds padding around the face, flips BGR→RGB,
        # and returns a PIL Image — the format PyTorch transforms expect.
        face_pil = self.embedder.crop_face_from_frame(frame, bbox)

        # ── Step 2: Embed ──
        # get_embedding() runs the PIL image through DINOv2 and returns a 1D tensor of shape (768,).
        # This 768-number vector is a mathematical "fingerprint" of the face.
        # Similar faces → similar vectors. Identical person → nearly identical vector.
        # The vector is L2-normalized (length = 1.0), so dot product = cosine similarity.
        embedding = self.embedder.get_embedding(face_pil)

        # ── Step 3: Match ──
        # find_match() searches the FAISS index for the nearest stored embedding.
        # Internally: reshape query to (1, 768) → index.search() → unwrap result → threshold check.
        # Returns:
        #   person_id  — SQLite integer ID of the best match (or None if below threshold)
        #   similarity — float 0.0–1.0: how similar the match embedding is to this face
        person_id, similarity = self.db.find_match(embedding, self.match_threshold)

        if person_id is not None:
            # ── KNOWN PERSON ──
            # get_person() looks up the PersonRecord from the in-memory dict — fast, no SQL query.
            # PersonRecord is a dataclass with: name, person_id, embeddings list, visit_count, timestamps.
            # person = self.db.get_person(person_id)

            # ── Cooldown check ──
            # dict.get(key, default) returns the stored value, or `default` if the key is absent.
            # Using 0 as default: "never greeted" → now - 0 is always huge → first greet always fires.
            last = self._last_greet.get(person_id, 0)
            if now - last >= self.greet_cooldown:
                # ── Greet via TTS ──
                # greeting_context = self.activity.get_greeting_context(person_id)
                # self.greeter.greet(GreetingEvent(
                #     person_name=person.name,
                #     visit_count=person.visit_count + 1,
                #     notes=greeting_context,
                #     similarity=similarity,
                #     timestamp=datetime.now(),
                # ))

                # ── Log sighting ──
                # record_sighting() writes to the SQLite sightings table and increments visit_count.
                # Called after greeter.greet() so the greeting fires without waiting for the DB write.
                self.db.record_sighting(person_id, similarity)

                # ── Auto-log visit context ──
                # datetime.now() gives a full datetime object.
                # .hour gives the hour as 0–23. The chained ternary maps that to a human label.
                dt = datetime.now()
                self.activity.log_sighting_context(person_id, {
                    # if dt.hour < 12 → "morning", elif < 17 → "afternoon", else → "evening"
                    "time_of_day": "morning" if dt.hour < 12 else "afternoon" if dt.hour < 17 else "evening",
                    # .strftime("%A") formats the datetime as the full weekday name: "Monday", "Tuesday", etc.
                    "day_of_week": dt.strftime("%A"),
                })

                # Mark this person as greeted right now.
                # Next frame: now - this value < greet_cooldown → cooldown not passed → skip greeting.
                self._last_greet[person_id] = now
        else:
            # ── UNKNOWN PERSON ──
            # No match above threshold — this face is not in the database.
            # Throttle saves: only queue a new unknown face once every unknown_cooldown seconds.
            if now - self._last_unknown_save >= self.unknown_cooldown:
                # queue_unlabeled() saves the face image (.png) and embedding (.pt) to disk,
                # and inserts a row into the SQLite unlabeled table for later human review.
                self.db.queue_unlabeled(face_pil, embedding)
                self._last_unknown_save = now  # reset the throttle clock



    # ─────────────────────────────────────────
    # DRAWING
    # ─────────────────────────────────────────
    def _draw_frame(self, frame, detections, fps):
        # Draw a bounding box and confidence label for each detected face.
        for bbox, conf in detections:
            # bbox is [x1, y1, x2, y2] as floats. OpenCV drawing functions require integers
            # because they refer to specific pixel positions (you can't address half a pixel).
            # List comprehension converts all four values at once: [int(c) for c in bbox].
            x1, y1, x2, y2 = [int(c) for c in bbox]

            # cv2.rectangle(image, top_left_corner, bottom_right_corner, color_BGR, line_thickness)
            # color (0, 255, 0) = green in BGR (Blue=0, Green=255, Red=0).
            # thickness=2 = a 2-pixel-wide border line.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # cv2.putText(image, text, origin_xy, font, font_scale, color_BGR, thickness)
            # origin = (x1, y1 - 8): place text 8 pixels above the top-left corner of the box.
            # f"{conf:.1%}" formats the float as a percentage with 1 decimal: 0.873 → "87.3%".
            # cv2.FONT_HERSHEY_SIMPLEX is a clean sans-serif font built into OpenCV (no install needed).
            # fontScale=0.5 makes it small; thickness=1 is a thin render.
            cv2.putText(frame, f"{conf:.1%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw the FPS counter in the top-left corner of the frame.
        # (10, 30) = 10 pixels from the left edge, 30 pixels from the top.
        # color (255, 255, 255) = white in BGR.
        # fontScale=0.7 and thickness=2 make it larger and bolder than the face confidence labels.
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw a database summary line in the bottom-left corner.
        stats = self.db.stats()
        info = f"Known: {stats['known_persons']} | Queue: {stats['unlabeled_queue']}"
        # frame.shape returns (height, width, channels). frame.shape[0] = height in pixels.
        # Subtracting 10 places the text 10 pixels above the bottom edge of the frame.
        cv2.putText(frame, info, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


    # ─────────────────────────────────────────
    # CLI LABELING (press 'l')
    # ─────────────────────────────────────────
    def _label_cli(self):
        # get_unlabeled() queries the SQLite unlabeled table for faces not yet reviewed.
        # Returns a list of dicts — one per face — with keys: id, image_path, embedding_path, timestamp.
        unlabeled = self.db.get_unlabeled()
        if not unlabeled:
            print("\n  ✨ No unlabeled faces!\n")
            return

        print(f"\n{'='*40}")
        print(f"  LABELING — {len(unlabeled)} faces")
        print(f"{'='*40}")

        for item in unlabeled:
            # cv2.imread() loads an image file from disk into a numpy array (BGR, uint8).
            # Returns None if the file path doesn't exist or the file can't be decoded.
            face_img = cv2.imread(item["image_path"])
            if face_img is not None:
                cv2.imshow("Label this face", face_img)
                # cv2.waitKey(500): pause for 500 milliseconds (0.5 seconds).
                # This gives the OS time to actually render the window before the terminal
                # steals focus for the input() prompt. Without this, the window may stay blank.
                cv2.waitKey(500)

            print(f"\n  Face from {item['timestamp']}")
            # input() pauses the program and waits for the user to type something + press Enter.
            # .strip() removes any leading/trailing whitespace from their input.
            name = input("  Name (or 'skip'/'discard'): ").strip()

            if name.lower() == "skip":
                continue  # move to the next face without doing anything to this one
            elif name.lower() == "discard":
                # Mark as reviewed=1 via direct SQL. This hides the face from future
                # get_unlabeled() calls without creating a person record for it.
                self.db.conn.execute(
                    "UPDATE unlabeled SET reviewed = 1 WHERE id = ?", (item["id"],)
                )
                # .commit() flushes the change from memory to disk.
                # SQLite wraps writes in a transaction — without commit(), changes are lost on crash.
                self.db.conn.commit()
            elif name:
                # label_face() does the full work: loads the stored embedding from disk,
                # creates or finds the person record, adds the embedding to FAISS, marks reviewed.
                self.db.label_face(item["id"], name)

        # cv2.destroyWindow() closes one specific named window, leaving all others open.
        # Compare to cv2.destroyAllWindows() which would close everything.
        cv2.destroyWindow("Label this face")
        print()

    # ─────────────────────────────────────────
    # GRADIO LABELING UI (press 'g')
    # ─────────────────────────────────────────
    def _launch_gradio(self):
        # thread.is_alive() returns True if the thread object exists and is still running.
        # Guard: don't start a second Gradio server if one is already up on port 7860.
        if self._gradio_thread and self._gradio_thread.is_alive():
            print("\n  Gradio already running at http://localhost:7860\n")
            return

        # Define the function the thread will execute.
        # This is a closure — run_gradio() can access `self` from the enclosing method
        # even after _launch_gradio() has already returned and its stack frame is gone.
        def run_gradio():
            try:
                # Import inside the thread function so an ImportError (gradio not installed)
                # is caught here in the thread, rather than crashing the whole program at startup.
                from labeling_ui import LabelingUI
                ui = LabelingUI(self.db)
                # ui.launch() starts a blocking web server inside this thread.
                # share=False: keep the UI local only (no public Gradio cloud URL).
                # server_port=7860: accessible at http://localhost:7860 in your browser.
                # quiet=True: suppress Gradio's startup banner from the terminal.
                ui.launch(share=False, server_port=7860, quiet=True)
            except ImportError:
                print(" Install gradio: pip install gradio")
            except Exception as e:
                print(f"  Gradio error: {e}")

        # threading.Thread wraps a function to run it concurrently in its own thread.
        # target=run_gradio — the function to call when the thread starts.
        #
        # daemon=True — marks this as a "daemon thread". What that means:
        #   Normally, Python waits for ALL threads to finish before the process exits.
        #   That would mean pressing 'q' to quit Lumemoria would not actually stop
        #   the process — the Gradio server thread would keep it alive.
        #   daemon=True tells Python: "when the main thread exits, kill this thread too."
        #   This gives us a clean exit without needing to manually stop the server.
        self._gradio_thread = threading.Thread(target=run_gradio, daemon=True)

        # .start() launches the thread. The OS schedules run_gradio() to begin executing
        # in the background. The main camera loop does NOT pause — both run at the same time.
        self._gradio_thread.start()
        print("\n  🌐 Gradio UI: http://localhost:7860\n")

    # ─────────────────────────────────────────
    # ACTIVITY NOTES (press 'a')
    # ─────────────────────────────────────────

    def _add_activity_note(self):
        # input() pauses and waits for the user to type. .strip() removes surrounding whitespace.
        name = input("\n  Person name: ").strip()
        if not name:
            return

        # self.db.conn is the raw sqlite3.Connection object.
        # .execute() sends a SQL query to the database.
        # The ? is a parameter placeholder — SQLite substitutes the value from the tuple.
        # Using ? (parameterized queries) prevents SQL injection:
        #   Never build a SQL string like f"WHERE name = '{name}'" — a user could type
        #   malicious SQL into the input and corrupt the database.
        # (name,) — the trailing comma makes this a tuple, not just parentheses around a string.
        row = self.db.conn.execute(
            "SELECT person_id FROM persons WHERE name = ?", (name,)
        ).fetchone()
        # .fetchone() returns a single result row as a tuple, e.g. (7,), or None if no match.

        if not row:
            print(f" '{name}' not found")
            return

        # row[0] extracts the first (only) column from the result tuple: the integer person_id.
        person_id = row[0]
        note = input("  Activity note: ").strip()
        if note:
            # activity.log() inserts a row into the activity_log table.
            # source="manual" distinguishes human-entered notes from auto-logged sighting context.
            self.activity.log(person_id, note, source="manual")
            # get_summary() reads the activity log and generates a plain-English summary string.
            summary = self.activity.get_summary(person_id)
            # update_notes() caches the summary in the persons table so the greeting system
            # can access it without re-summarizing the full log on every frame.
            self.db.update_notes(person_id, summary)
            print(f" Saved. Summary: {summary}")
        print()

    # ─────────────────────────────────────────
    # PERSON DETAILS (press 'p')
    # ─────────────────────────────────────────

    def _show_person_details(self):
        name = input("\n  Person name: ").strip()
        if not name:
            return

        # Same SQL lookup pattern as _add_activity_note above.
        row = self.db.conn.execute(
            "SELECT person_id FROM persons WHERE name = ?", (name,)
        ).fetchone()

        if not row:
            print(f"  ❌ '{name}' not found")
            return

        person_id = row[0]
        # get_person() returns the PersonRecord dataclass from the in-memory dict — no SQL query.
        # PersonRecord.embeddings is the list of all stored embedding tensors for this person.
        person = self.db.get_person(person_id)
        # get_visit_patterns() analyzes the sightings log to compute:
        # average visits per week, most common day of week, most common time of day.
        patterns = self.activity.get_visit_patterns(person_id)
        # get_recent() returns the last N activity log entries as a list of log entry objects.
        recent_notes = self.activity.get_recent(person_id, limit=5)
        summary = self.activity.get_summary(person_id)

        print(f"\n  {'─'*40}")
        print(f"  {person.name} (ID: {person.person_id})")
        print(f"  {'─'*40}")
        print(f"  Embeddings: {len(person.embeddings)}")
        print(f"  Visits:     {person.visit_count}")
        # person.first_seen and .last_seen are ISO datetime strings: "2026-02-10T09:30:00".
        # [:10] slices the first 10 characters → just the date: "2026-02-10".
        # The conditional expression handles None (no sightings yet) by showing a dash instead.
        print(f"  First seen: {person.first_seen[:10] if person.first_seen else '—'}")
        print(f"  Last seen:  {person.last_seen[:10] if person.last_seen else '—'}")

        if patterns.get("total_visits", 0) > 0:
            print(f"\n  Visit patterns:")
            print(f"    Avg/week:  {patterns.get('avg_visits_per_week', 0)}")
            print(f"    Best day:  {patterns.get('most_common_day', '—')}")
            print(f"    Best time: {patterns.get('most_common_time', '—')}")

        if recent_notes:
            print(f"\n  Recent notes:")
            for entry in recent_notes:
                # entry.source = "manual" or "auto". entry.note = the text content of the log entry.
                print(f"    [{entry.source}] {entry.note}")

        print(f"\n  Summary: {summary}")
        print()

    # ─────────────────────────────────────────
    # STATS (press 's')
    # ─────────────────────────────────────────

    def _print_stats(self):
        # db.stats() returns a dict with current counts pulled from SQLite and the FAISS index:
        #   known_persons, total_embeddings, unlabeled_queue, faiss_index_type, faiss_index_size
        stats = self.db.stats()
        print(f"\n  📊 Database:")
        print(f"    Persons: {stats['known_persons']}")
        print(f"    Embeddings: {stats['total_embeddings']}")
        print(f"    Unlabeled: {stats['unlabeled_queue']}")
        # .get() with defaults: if these keys are absent, print gracefully instead of raising KeyError.
        print(f"    FAISS: {stats.get('faiss_index_type', 'N/A')} ({stats.get('faiss_index_size', 0)} vectors)")

        # get_all_persons() returns a list of all PersonRecord objects from the in-memory dict.
        for person in self.db.get_all_persons():
            summary = self.activity.get_summary(person.person_id)
            print(f"    • {person.name} — {summary}")
        print()


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

# This block only runs when you execute this file directly: python pipeline.py
# If another file imports pipeline.py (e.g. for testing), this block is skipped.
# __name__ is a special Python variable:
#   "__main__" → this file was run directly
#   "pipeline"  → this file was imported by something else
if __name__ == "__main__":
    app = Lumemoria(
        yolo_model_path="yolov8n-face.pt",
        dinov3_model="dinov2_vitb14",
        db_path="./lumemoria_data",
        detection_conf=0.5,
        match_threshold=0.65,
        greet_cooldown=300,
        unknown_cooldown=30,
        tts_backend="auto",
    )
    app.run(camera_index=0)
