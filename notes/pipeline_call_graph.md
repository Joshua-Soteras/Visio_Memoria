# pipeline.py — Lumemoria Function Call Graph

How every function inside `Lumemoria` connects to every other function.
External modules (FaceEmbedder, FaceDatabase, Greeter, ActivityTracker) are shown
as black boxes — their internals are not expanded here.

---

## Legend

```
[Method]         ← Lumemoria method
[ Module ]       ← external module (black box)
key: 'x'         ← triggered by keyboard input inside run()
──►              ← calls / dispatches to
loop             ← called repeatedly every frame
once             ← called once
```

---

## Master Call Graph — All Relationships

`run()` is the central hub. Every other method is either called by `run()`
directly or triggered by a keypress caught inside `run()`.

```
                          __init__()
                              │ once
                    ┌─────────┼─────────────────┐
                    ▼         ▼                  ▼
              [YOLO]    [FaceEmbedder]     [FaceDatabase]
                              │                  │
                              ▼                  ▼
                         [Greeter]       [ActivityTracker]


                            run()
                              │
         ┌────────────────────┼──────────────────────────────┐
         │        frame loop  │                              │
         │  ┌─────────────────┤                              │
         │  │                 │                              │
         ▼  ▼                 ▼                              ▼
  _extract_           _process_face()              keypress dispatch
  detections()         (per face)                        │
                                              ┌───────────┼───────────────┬────────────────┬───────────────┐
                                              ▼           ▼               ▼                ▼               ▼
                                          key:'l'      key:'g'         key:'s'          key:'a'         key:'p'
                                              │           │               │                │               │
                                              ▼           ▼               ▼                ▼               ▼
                                        _label_cli  _launch_gradio  _print_stats  _add_activity   _show_person
                                                                                     _note()         _details()

         │
         ▼
    _draw_frame()
    (every frame, after _process_face)
```

---

## Scenario 1 — Startup (`__init__`)

What fires when `Lumemoria()` is created.

```
__init__()
  │
  ├─1─► YOLO("yolov8n-face.pt")       → self.detector
  │
  ├─2─► FaceEmbedder(model_name)      → self.embedder
  │
  ├─3─► FaceDatabase(db_path)         → self.db
  │       └── self.db.load_faiss_index()
  │             if False: print fallback message
  │
  ├─4─► Greeter(backend)              → self.greeter
  │       └── self.greeter.start()
  │
  ├─5─► ActivityTracker(self.db.conn) → self.activity
  │
  └─6─► self.db.stats()               → print summary to console
```

---

## Scenario 2 — The Frame Loop (`run`)

The core of the pipeline. Runs continuously until 'q' is pressed.

```
run(camera_index=0)
  │
  ├── cv2.VideoCapture(camera_index)
  │
  └── while True:
        │
        ├─1─► cap.read()                     → frame (raw BGR image)
        │
        ├─2─► self.detector(frame)            → YOLO results
        │       │
        │       └─► _extract_detections(results)
        │                 │
        │                 └── returns: detections [ (bbox, conf), ... ]
        │                             landmarks_list [ kps, ... ]
        │
        ├─3─► for each (bbox, conf) in detections:
        │         │
        │         └─► _process_face(frame, bbox, conf, landmarks)
        │
        ├─4─► _draw_frame(frame, detections, fps_display)
        │
        ├─5─► cv2.imshow(frame)
        │
        ├─6─► cv2.waitKey(1) → key
        │         │
        │         ├── 'q' → break
        │         ├── 'l' → _label_cli()
        │         ├── 'g' → _launch_gradio()
        │         ├── 's' → _print_stats()
        │         ├── 'a' → _add_activity_note()
        │         └── 'p' → _show_person_details()
        │
        └── [FPS counter updated once per second]

  finally:  ← runs on 'q' OR on any exception
        ├── self.greeter.stop()
        ├── self.db.save_faiss_index()
        ├── cap.release()
        └── cv2.destroyAllWindows()
```

---

## Scenario 3 — Processing One Face (`_process_face`)

Called once per detected face per frame. The core recognition logic.

```
_process_face(frame, bbox, conf, landmarks)
  │
  ├─1─► self.embedder.crop_face_from_frame(frame, bbox)
  │           → face_pil  (PIL Image, cropped + padded)
  │
  ├─2─► self.embedder.get_embedding(face_pil)
  │           → embedding  (Tensor, shape (768,), L2-normalized)
  │
  ├─3─► self.db.find_match(embedding, self.match_threshold)
  │           → (person_id, similarity)  OR  (None, score)
  │
  │
  ├── person_id is NOT None?  (KNOWN PERSON)
  │       │
  │       ├── self.db.get_person(person_id)            → person (PersonRecord)
  │       │
  │       ├── cooldown check: now - _last_greet[person_id] >= greet_cooldown?
  │       │       │
  │       │       └── YES (cooldown passed):
  │       │               │
  │       │               ├─► self.activity.get_greeting_context(person_id)
  │       │               │         → greeting_context (string of notes)
  │       │               │
  │       │               ├─► self.greeter.greet(GreetingEvent(...))
  │       │               │         speaks name + visit count via TTS
  │       │               │
  │       │               ├─► self.db.record_sighting(person_id, similarity)
  │       │               │         logs to SQLite, increments visit_count
  │       │               │
  │       │               ├─► self.activity.log_sighting_context(person_id, {...})
  │       │               │         auto-logs time_of_day + day_of_week
  │       │               │
  │       │               └── self._last_greet[person_id] = now
  │
  └── person_id is None?  (UNKNOWN PERSON)
          │
          └── cooldown check: now - _last_unknown_save >= unknown_cooldown?
                  │
                  └── YES:
                          ├─► self.db.queue_unlabeled(face_pil, embedding)
                          │         saves .png + .pt, inserts SQLite row
                          └── self._last_unknown_save = now
```

---

## Scenario 4 — Keypress Handlers

Each key handler and its internal steps.

```
key 'l' → _label_cli()
  │
  ├── self.db.get_unlabeled()                → list of unreviewed faces
  │
  └── for each unlabeled face:
        ├── cv2.imshow("Label this face", ...)
        ├── input("Name or skip/discard")
        │
        ├── 'skip'    → continue (leave as unreviewed)
        ├── 'discard' → self.db.conn.execute("UPDATE unlabeled SET reviewed=1 ...")
        └── name      → self.db.label_face(item["id"], name)


key 'g' → _launch_gradio()
  │
  ├── check: self._gradio_thread already alive? → print message, return
  │
  └── spawn daemon thread:
        └── run_gradio():
              └── LabelingUI(self.db).launch(port=7860)


key 's' → _print_stats()
  │
  ├── self.db.stats()                         → dict (persons, embeddings, queue, FAISS info)
  │
  └── for each person in self.db.get_all_persons():
          └── self.activity.get_summary(person_id) → string


key 'a' → _add_activity_note()
  │
  ├── input("Person name")
  ├── self.db.conn.execute("SELECT person_id FROM persons WHERE name=?")
  ├── input("Activity note")
  ├── self.activity.log(person_id, note, source="manual")
  ├── self.activity.get_summary(person_id)    → updated summary string
  └── self.db.update_notes(person_id, summary)


key 'p' → _show_person_details()
  │
  ├── input("Person name")
  ├── self.db.conn.execute("SELECT person_id FROM persons WHERE name=?")
  ├── self.db.get_person(person_id)           → PersonRecord
  ├── self.activity.get_visit_patterns(person_id)
  ├── self.activity.get_recent(person_id, limit=5)
  └── self.activity.get_summary(person_id)
```

---

## Scenario 5 — Drawing (`_draw_frame`)

Called every frame after all faces are processed.

```
_draw_frame(frame, detections, fps)
  │
  ├── for each (bbox, conf) in detections:
  │     ├── cv2.rectangle(frame, ...)       draw green box around face
  │     └── cv2.putText(frame, conf, ...)   draw confidence score above box
  │
  ├── cv2.putText(frame, "FPS: N", ...)     top-left corner
  │
  └── self.db.stats()
        └── cv2.putText(frame, "Known: N | Queue: N", ...)  bottom-left corner
```

---

## Who Calls Who — Quick Reference Table

| Method | Called by | Calls internally |
|---|---|---|
| `__init__` | external (`Lumemoria()`) | `self.db.load_faiss_index`, `self.greeter.start`, `self.db.stats` |
| `run` | external (`.run()`) | `_extract_detections`, `_process_face`, `_draw_frame`, `_label_cli`, `_launch_gradio`, `_print_stats`, `_add_activity_note`, `_show_person_details` |
| `_extract_detections` | `run` | nothing |
| `_process_face` | `run` (once per face) | nothing |
| `_draw_frame` | `run` (once per frame) | nothing |
| `_label_cli` | `run` (key 'l') | nothing |
| `_launch_gradio` | `run` (key 'g') | spawns thread → `LabelingUI` |
| `_print_stats` | `run` (key 's') | nothing |
| `_add_activity_note` | `run` (key 'a') | nothing |
| `_show_person_details` | `run` (key 'p') | nothing |

**Key observation:** `run()` is the only method that calls other Lumemoria methods.
All other methods are leaf nodes — they call out to external modules but never
call back into each other. This keeps the pipeline simple and predictable.
