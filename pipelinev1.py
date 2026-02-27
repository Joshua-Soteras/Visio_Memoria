"""
pipelinev1.py — Lumemoria Basic Testing Pipeline (v1)
Stripped-down version of pipeline.py for testing core functionality.

What this version does:
  YOLOv8-face → DINOv2 → FAISS → terminal print (known) / queue (unknown)

What was removed vs pipeline.py:
  - Greeter (TTS) — removed entirely
  - ActivityTracker — removed entirely
  - Gradio labeling UI ('g' key) — removed
  - Activity note ('a' key) — removed
  - Person details ('p' key) — removed (depended on ActivityTracker)
  - threading import — no longer needed without Gradio
  - tts_backend parameter — no longer needed without Greeter
  - greet_cooldown renamed to print_cooldown (now controls terminal print rate, not TTS)

What was added vs pipeline.py:
  - Terminal print when a known person is matched (replaces TTS greeting)
  - _draw_dashed_rect() — dashed/dither style bounding box

Testing improvements added in this revision:
  [IMPROVEMENT 1] Color-coded bounding boxes — green=known, yellow=unknown
  [IMPROVEMENT 2] Name + similarity score drawn directly on frame
  [IMPROVEMENT 3] record_sighting decoupled from print cooldown (own 1s cooldown)
  [IMPROVEMENT 4] Lowered default cooldowns for faster testing feedback
  [IMPROVEMENT 5] stats() cached — SQLite queried once per second, not every frame
  [IMPROVEMENT 6] Debug key 'd' — dumps full system state to terminal

Run:   python pipelinev1.py
Quit:  press 'q'
Label: press 'l' (CLI)
Stats: press 's'
Debug: press 'd'
"""

import cv2
import time
from datetime import datetime

# FaceEmbedder — crops face regions from frames and runs DINOv2 to produce
# 768-dimensional embedding vectors representing each face's identity.
from src.visio_memoria.models.FaceEmbedder import FaceEmbedder

# FaceDatabaseFAISS — SQLite + FAISS-backed store for face embeddings.
# Handles known person lookup, sighting logging, and the unlabeled queue.
from src.visio_memoria.models.face_database_faiss import FaceDatabaseFAISS as FaceDatabase

import os
from dotenv import load_dotenv

# Get the current working directory
root_dir = os.getcwd()

# Join the absolute root directory with the target file path
DEFAULT_YOLO_MODEL = os.path.join(root_dir, "src", "visio_memoria", "models", "yolov8-face", "yolov8n-face.pt")
DEFAULT_DB_PATH = os.path.join(root_dir, "src", "visio_memoria", "db_paths")



class VisioMemoria:
    """
    Basic recognition pipeline for testing.

    Frame-by-frame flow:
      1. YOLOv8-face → bounding boxes + landmarks
      2. Crop faces → PIL images
      3. DINOv2 → 768-dim embeddings
      4. FAISS → match against database
      5a. Known   → print name to terminal + log sighting to SQLite
      5b. Unknown → queue face image + embedding to disk for later labeling
    """

    def __init__(
        self,
        yolo_model_path: str = DEFAULT_YOLO_MODEL,
        dinov3_model: str = "dinov3_vitb14",
        db_path: str = DEFAULT_DB_PATH,
        detection_conf: float = 0.5,
        match_threshold: float = 0.65,
        # IMPROVEMENT 4 — LOWERED FROM 10s TO 3s FOR FASTER TESTING FEEDBACK
        print_cooldown: int = 3,
        # IMPROVEMENT 4 — LOWERED FROM 30s TO 5s FOR FASTER QUEUE TESTING
        unknown_cooldown: int = 5,
        # IMPROVEMENT 3 — NEW PARAM: HOW OFTEN TO RECORD A SIGHTING TO SQLITE,
        # SEPARATE FROM THE PRINT COOLDOWN SO VISIT COUNTS ARE ACCURATE
        sighting_cooldown: int = 1,
    ):
        # Lazy import — ultralytics is heavy; only load it when the pipeline is created.
        from ultralytics import YOLO

        print("=" * 50)
        print("  Visio Memoria  v1 — Basic Test Pipeline")
        print("=" * 50)

        # ── 1. Detection ──
        print("\n[1/3] Loading YOLOv8-face...")
        self.detector = YOLO(yolo_model_path)
        self.detection_conf = detection_conf

        # ── 2. Embedding ──
        # empty parameters means default model will be used
        print("[2/3] Loading DINOv3...")
        self.embedder = FaceEmbedder()

        # ── 3. Database (FAISS) ──
        print("[3/3] Loading face database (FAISS)...")
        self.db = FaceDatabase(db_path)

        # Try loading a saved FAISS index from disk — faster than rebuilding from embeddings.
        # Returns False on first run (no saved index yet); FaceDatabase already rebuilt it.
        if not self.db.load_faiss_index():
            print("  No saved index, rebuilt from embeddings")

        # ── Config ──
        self.match_threshold = match_threshold
        self.print_cooldown = print_cooldown
        self.unknown_cooldown = unknown_cooldown
        # IMPROVEMENT 3 — STORE SIGHTING COOLDOWN AS INSTANCE VAR
        self.sighting_cooldown = sighting_cooldown

        # ── Cooldown trackers ──
        # Maps person_id → timestamp of last terminal print for that person.
        # dict.get(person_id, 0) returns 0 (Unix epoch) if never printed → always fires first time.
        self._last_print: dict[int, float] = {}

        # IMPROVEMENT 3 — SEPARATE SIGHTING TRACKER, INDEPENDENT OF PRINT COOLDOWN.
        # ALLOWS VISIT COUNTS TO BE RECORDED MORE FREQUENTLY THAN TERMINAL PRINTS.
        self._last_sighting: dict[int, float] = {}

        # Timestamp of the last unknown face saved to the queue.
        self._last_unknown_save: float = 0

        # IMPROVEMENT 5 — STATS CACHE: STORES THE LAST DB QUERY RESULT SO WE DON'T
        # HIT SQLITE ON EVERY SINGLE FRAME (30 QUERIES/SEC REDUCED TO 1 QUERY/SEC)
        self._cached_stats: dict = {}
        self._stats_cache_time: float = 0

        stats = self.db.stats()
        print(f"\n  Known: {stats['known_persons']} persons, "
              f"{stats['total_embeddings']} embeddings, "
              f"FAISS: {stats.get('faiss_index_type', 'N/A')}")
        print(f"  Unlabeled queue: {stats['unlabeled_queue']}")
        print("\n✅ Visio Memoria v1 ready!\n")



    # ─────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────
    def run(self, camera_index: int = 0):
        """Main camera loop."""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return

        print("Controls:")
        print("  q → quit")
        print("  l → label faces (CLI)")
        print("  s → stats")
        # IMPROVEMENT 6 — ADVERTISE DEBUG KEY IN STARTUP CONTROLS
        print("  d → debug dump")
        print()

        fps_counter = 0
        fps_time = time.time()
        fps_display = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── Detect ──
                results = self.detector(frame, verbose=False)
                detections, landmarks_list = self._extract_detections(results)

                # IMPROVEMENT 1/2 — COLLECT PER-FACE MATCH RESULTS SO _draw_frame()
                # KNOWS WHETHER EACH FACE IS KNOWN OR UNKNOWN BEFORE DRAWING
                face_results = []
                for i, (bbox, conf) in enumerate(detections):
                    landmarks = landmarks_list[i] if i < len(landmarks_list) else None
                    result = self._process_face(frame, bbox, conf, landmarks)
                    face_results.append(result)

                # ── Draw ──
                # IMPROVEMENT 1/2 — PASS face_results TO _draw_frame FOR COLOR-CODING
                self._draw_frame(frame, detections, face_results, fps_display)

                # ── FPS ──
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_time = time.time()

                # ── Display ──
                cv2.imshow("Visio Memoria — v1", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("l"):
                    self._label_cli()
                elif key == ord("s"):
                    self._print_stats()
                # IMPROVEMENT 6 — DEBUG KEY: PRESS 'd' TO DUMP SYSTEM STATE
                elif key == ord("d"):
                    self._print_debug()

        finally:
            self.db.save_faiss_index()
            cap.release()
            cv2.destroyAllWindows()
            print("\nVisio Memoria v1 stopped. FAISS index saved.")



    # ─────────────────────────────────────────
    # DETECTION
    # ─────────────────────────────────────────
    def _extract_detections(self, results) -> tuple[list, list]:
        """Extract bboxes, confidences, and landmarks from YOLO results."""
        detections = []
        landmarks_list = []

        for result in results:
            if result.boxes is None:
                continue

            for i, box in enumerate(result.boxes):
                # box.conf[0].item() → plain Python float confidence score
                conf = box.conf[0].item()

                if conf >= self.detection_conf:
                    # box.xyxy[0].tolist() → [x1, y1, x2, y2] as floats
                    bbox = box.xyxy[0].tolist()
                    detections.append((bbox, conf))

                    # Landmarks: 5 facial keypoints (eyes, nose, mouth corners).
                    # Passed through but not yet used — reserved for future face alignment.
                    if hasattr(result, "keypoints") and result.keypoints is not None:
                        try:
                            kps = result.keypoints.xy[i].cpu().numpy()
                            landmarks_list.append(kps)
                        except (IndexError, AttributeError):
                            landmarks_list.append(None)
                    else:
                        landmarks_list.append(None)

        return detections, landmarks_list



    # ─────────────────────────────────────────
    # CORE: PROCESS A SINGLE FACE
    # ─────────────────────────────────────────
    def _process_face(self, frame, bbox, conf, landmarks=None) -> dict:
        """Crop → embed → match → print or queue.

        IMPROVEMENT 1/2 — NOW RETURNS A DICT WITH MATCH RESULT SO _draw_frame()
        CAN COLOR-CODE THE BOX AND SHOW THE NAME/SIMILARITY ON FRAME.
        """
        now = time.time()

        # ── Step 1: Crop ──
        # Crops the face region from the BGR frame, adds padding, flips to RGB,
        # and returns a PIL Image ready for DINOv2.
        face_pil = self.embedder.crop_face_from_frame(frame, bbox)

        # ── Step 2: Embed ──
        # Returns a 768-dim L2-normalized tensor — the face's mathematical fingerprint.
        embedding = self.embedder.get_embedding(face_pil)

        # ── Step 3: Match ──
        # FAISS searches for the nearest stored embedding.
        # Returns (person_id, similarity) or (None, best_similarity) if below threshold.
        person_id, similarity = self.db.find_match(embedding, self.match_threshold)

        if person_id is not None:
            # ── KNOWN PERSON ──
            person = self.db.get_person(person_id)

            # IMPROVEMENT 3 — RECORD SIGHTING ON ITS OWN 1-SECOND COOLDOWN, SEPARATE
            # FROM THE PRINT COOLDOWN. THIS MEANS VISIT COUNTS ACCUMULATE ACCURATELY
            # EVEN IF THE TERMINAL PRINT IS SUPPRESSED BY THE LONGER PRINT_COOLDOWN.
            if now - self._last_sighting.get(person_id, 0) >= self.sighting_cooldown:
                self.db.record_sighting(person_id, similarity)
                self._last_sighting[person_id] = now

            # IMPROVEMENT 4 — PRINT COOLDOWN NOW 3s (WAS 10s) FOR FASTER FEEDBACK
            last = self._last_print.get(person_id, 0)
            if now - last >= self.print_cooldown:
                print(f"  Recognized: {person.name}  (similarity={similarity:.2f}, visits={person.visit_count})")
                self._last_print[person_id] = now

            # IMPROVEMENT 1/2 — RETURN KNOWN RESULT FOR COLOR-CODED BOX AND ON-FRAME LABEL
            return {"known": True, "name": person.name, "similarity": similarity}

        else:
            # ── UNKNOWN PERSON ──
            # IMPROVEMENT 4 — UNKNOWN COOLDOWN NOW 5s (WAS 30s) FOR FASTER QUEUE TESTING
            if now - self._last_unknown_save >= self.unknown_cooldown:
                self.db.queue_unlabeled(face_pil, embedding)
                self._last_unknown_save = now

            # IMPROVEMENT 1/2 — RETURN UNKNOWN RESULT. SIMILARITY STILL SHOWN SO YOU CAN
            # SEE HOW CLOSE THE BEST NON-MATCH WAS — USEFUL FOR TUNING match_threshold.
            return {"known": False, "name": None, "similarity": similarity}



    # ─────────────────────────────────────────
    # DRAWING
    # ─────────────────────────────────────────
    def _draw_dashed_rect(self, frame, x1, y1, x2, y2, color, thickness=1, dash=8, gap=5):
        """Draw a dashed (dither-style) rectangle.

        OpenCV has no native dashed line, so we manually walk each side in
        (dash + gap) pixel steps, drawing only the dash segment each time.
        """
        # Top and bottom sides (horizontal)
        for y in [y1, y2]:
            x = x1
            while x < x2:
                x_end = min(x + dash, x2)
                cv2.line(frame, (x, y), (x_end, y), color, thickness)
                x += dash + gap

        # Left and right sides (vertical)
        for x in [x1, x2]:
            y = y1
            while y < y2:
                y_end = min(y + dash, y2)
                cv2.line(frame, (x, y), (x, y_end), color, thickness)
                y += dash + gap

    def _draw_frame(self, frame, detections, face_results, fps):
        """Draw bounding boxes, labels, FPS, and DB stats onto the frame."""
        for i, (bbox, _) in enumerate(detections):
            # bbox floats → ints (pixel coordinates)
            x1, y1, x2, y2 = [int(c) for c in bbox]

            result = face_results[i] if i < len(face_results) else {}

            # IMPROVEMENT 1 — COLOR-CODED BOXES: GREEN = KNOWN PERSON, YELLOW = UNKNOWN
            is_known = result.get("known", False)
            color = (0, 255, 0) if is_known else (0, 255, 255)

            self._draw_dashed_rect(frame, x1, y1, x2, y2, color, thickness=1)

            # IMPROVEMENT 2 — NAME + SIMILARITY SCORE DRAWN ON FRAME SO YOU CAN
            # VALIDATE RECOGNITION DECISIONS WITHOUT WATCHING THE TERMINAL.
            # KNOWN:   "Josh 0.82"
            # UNKNOWN: "Unknown 0.41" (shows best near-miss score for threshold tuning)
            if is_known:
                label = f"{result['name']} {result['similarity']:.2f}"
            else:
                label = f"Unknown {result.get('similarity', 0):.2f}"

            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # FPS counter — top-left corner
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # IMPROVEMENT 5 — CACHED STATS: ONLY QUERY SQLITE ONCE PER SECOND INSTEAD OF
        # EVERY FRAME. AT 30FPS THAT'S 30 DB READS/SEC AVOIDED.
        now = time.time()
        if now - self._stats_cache_time >= 1.0:
            self._cached_stats = self.db.stats()
            self._stats_cache_time = now

        # DB summary — bottom-left corner
        info = (f"Known: {self._cached_stats.get('known_persons', 0)} "
                f"| Queue: {self._cached_stats.get('unlabeled_queue', 0)}")
        cv2.putText(frame, info, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)



    # ─────────────────────────────────────────
    # CLI LABELING (press 'l')
    # ─────────────────────────────────────────
    def _label_cli(self):
        """Show each unlabeled face and prompt for a name."""
        unlabeled = self.db.get_unlabeled()
        if not unlabeled:
            print("\n  No unlabeled faces!\n")
            return

        print(f"\n{'='*40}")
        print(f"  LABELING — {len(unlabeled)} faces")
        print(f"{'='*40}")

        for item in unlabeled:
            # Load and show the saved face image in an OpenCV window.
            face_img = cv2.imread(item["image_path"])
            if face_img is not None:
                cv2.imshow("Label this face", face_img)
                # Wait 500ms so the window renders before the terminal steals focus.
                cv2.waitKey(500)

            print(f"\n  Face from {item['timestamp']}")
            name = input("  Name (or 'skip'/'discard'): ").strip()

            if name.lower() == "skip":
                continue
            elif name.lower() == "discard":
                # Mark reviewed=1 in SQLite — hides from future get_unlabeled() calls.
                self.db.conn.execute(
                    "UPDATE unlabeled SET reviewed = 1 WHERE id = ?", (item["id"],)
                )
                self.db.conn.commit()
            elif name:
                # label_face() loads the embedding, creates/updates the person record,
                # adds embedding to FAISS, and marks the unlabeled row as reviewed.
                self.db.label_face(item["id"], name)

        cv2.destroyWindow("Label this face")
        print()



    # ─────────────────────────────────────────
    # STATS (press 's')
    # ─────────────────────────────────────────
    def _print_stats(self):
        """Print database and FAISS index stats to the terminal."""
        stats = self.db.stats()
        print(f"\n  Database:")
        print(f"    Persons:    {stats['known_persons']}")
        print(f"    Embeddings: {stats['total_embeddings']}")
        print(f"    Unlabeled:  {stats['unlabeled_queue']}")
        print(f"    FAISS:      {stats.get('faiss_index_type', 'N/A')} "
              f"({stats.get('faiss_index_size', 0)} vectors)")

        # List each known person with their visit count.
        for person in self.db.get_all_persons():
            print(f"    • {person.name} — {person.visit_count} visit(s)")
        print()


    # ─────────────────────────────────────────
    # IMPROVEMENT 6 — DEBUG DUMP (press 'd')
    # PRINTS FULL SYSTEM STATE IN ONE KEYPRESS: FAISS STATUS, ALL KNOWN PERSONS,
    # COOLDOWN TIMERS, AND CURRENT THRESHOLD VALUES. USEFUL FOR DIAGNOSING WHY
    # A FACE ISN'T BEING RECOGNIZED OR WHY A SIGHTING ISN'T BEING LOGGED.
    # ─────────────────────────────────────────

    def _print_debug(self):
        """Dump full system state to terminal for diagnosing recognition issues."""
        stats = self.db.stats()
        now = time.time()

        print(f"\n{'='*40}")
        print(f"  DEBUG — {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*40}")
        print(f"  FAISS:        {stats.get('faiss_index_type', 'N/A')} "
              f"({stats.get('faiss_index_size', 0)} vectors)")
        print(f"  Persons:      {stats['known_persons']}")
        print(f"  Embeddings:   {stats['total_embeddings']}")
        print(f"  Unlabeled:    {stats['unlabeled_queue']}")
        print(f"  Thresholds:")
        print(f"    match_threshold:   {self.match_threshold}")
        print(f"    print_cooldown:    {self.print_cooldown}s")
        print(f"    sighting_cooldown: {self.sighting_cooldown}s")
        print(f"    unknown_cooldown:  {self.unknown_cooldown}s")
        print(f"  Known persons:")
        for person in self.db.get_all_persons():
            last_print = self._last_print.get(person.person_id, 0)
            last_sight = self._last_sighting.get(person.person_id, 0)
            print_since = f"{now - last_print:.1f}s ago" if last_print > 0 else "never"
            sight_since = f"{now - last_sight:.1f}s ago" if last_sight > 0 else "never"
            print(f"    • {person.name} — {person.visit_count} visit(s) | "
                  f"last print: {print_since} | last sighting: {sight_since}")
        print()


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    app = VisioMemoria(
        yolo_model_path=DEFAULT_YOLO_MODEL,
        db_path=DEFAULT_DB_PATH,
        detection_conf=0.5,
        match_threshold=0.65,
        print_cooldown=3,
        unknown_cooldown=5,
        sighting_cooldown=1,
    )
    app.run(camera_index=0)
