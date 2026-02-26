# face_database_faiss.py — Concepts & Lessons Learned

Notes on the concepts behind `FaceDatabaseFAISS`, written as a learning reference.
Each topic maps to something implemented in the code.

---

## The Big Picture — How Everything Connects

```
┌─────────────────────────────────────────────────────────┐
│                  FaceDatabaseFAISS                       │
│                                                          │
│  ┌──────────────────┐     ┌───────────────────────────┐ │
│  │  self.persons     │     │     FAISS Layer            │ │
│  │  dict[int,        │     │                            │ │
│  │  PersonRecord]    │     │  self._faiss_index         │ │
│  │                  │     │  (IndexFlatIP or            │ │
│  │  • person_id     │◄────│   IndexIVFFlat)             │ │
│  │  • name          │     │                            │ │
│  │  • embeddings[]  │     │  self._faiss_id_to_person_id│ │
│  │  • visit_count   │     │  list[int]  ← parallel map │ │
│  └──────────────────┘     └───────────────────────────┘ │
│            ▲                           ▲                 │
│            │                           │                 │
└────────────┼───────────────────────────┼─────────────────┘
             │                           │
    ┌────────┴──────────┐      ┌─────────┴──────────────┐
    │   SQLite          │      │   File System           │
    │  visio_memoria.db │      │                         │
    │                  │      │  embeddings/             │
    │  • persons        │      │    person_1.pt           │
    │  • sightings      │      │    person_2.pt ...       │
    │  • unlabeled      │      │                         │
    └───────────────────┘      │  unlabeled/             │
                               │    2026-02-25_face.png  │
                               │    2026-02-25_face.pt   │
                               │                         │
                               │  faiss_index.bin        │
                               │  faiss_mapping.json     │
                               └─────────────────────────┘
```

**Three separate systems must stay in sync at all times:**

| Layer | What it stores | Format |
|---|---|---|
| SQLite | Metadata (names, timestamps, visit counts) | Relational rows |
| File system (`.pt`) | Raw embedding tensors per person | PyTorch serialized |
| FAISS index | Flat array of all embeddings for fast search | Compressed binary |

The **parallel list** (`_faiss_id_to_person_id`) is the bridge between FAISS and SQLite.
FAISS only knows positions (0, 1, 2…). The parallel list translates those positions into real `person_id` values.

---

## The Two-Layer ID System

This is the most important concept in the file. FAISS has no concept of "people" — it only has rows.

```
FAISS internal storage (flat array):
  Row 0 → [0.12, -0.33, 0.87, ...]   ← embedding for Person A, photo 1
  Row 1 → [0.09, -0.31, 0.90, ...]   ← embedding for Person A, photo 2
  Row 2 → [0.55,  0.21, 0.04, ...]   ← embedding for Person B, photo 1

Parallel Python list  (_faiss_id_to_person_id):
  [0]  →  person_id = 7     (maps FAISS row 0 → SQLite person_id 7)
  [1]  →  person_id = 7     (same person, different photo)
  [2]  →  person_id = 12    (maps FAISS row 2 → SQLite person_id 12)
```

When `find_match()` returns FAISS row 2, we look up `_faiss_id_to_person_id[2]` = 12,
then query SQLite for person_id 12 to get the name, stats, etc.

**Both lists must grow together** — every `_faiss_index.add(emb)` call must be immediately
followed by `_faiss_id_to_person_id.append(person_id)`. If they get out of sync, the wrong
person gets identified.

---

## FAISS Pipeline — Function Call Map

### Startup (called once in `__init__`)

```
__init__()
  │
  ├─► _init_tables()          — create SQLite tables if they don't exist
  │
  ├─► _load_all()             — load every PersonRecord from SQLite + .pt files
  │         └─ reads: persons table, embeddings/person_N.pt
  │
  └─► _rebuild_faiss_index()  — build FAISS index from all loaded embeddings
            └─ decides: IndexFlatIP (< 5000 embs) or IndexIVFFlat (≥ 5000)
```

### Recognition (called every frame)

```
find_match(query_emb)
  │
  ├─► index.search(query_np, k=1)    — FAISS returns (distances, indices)
  │
  ├─► _faiss_id_to_person_id[idx]    — translate FAISS row → person_id
  │
  └─► return (person_id, similarity) — or (None, score) if below threshold
```

### New Face — Unknown Person

```
queue_unlabeled(face_image, embedding)
  │
  ├─► saves: unlabeled/timestamp_N.png   (human-viewable image)
  ├─► saves: unlabeled/timestamp_N.pt    (embedding for AI to reuse)
  └─► inserts row into SQLite unlabeled table
```

### Labeling an Unknown Face

```
label_face(unlabeled_id, name)
  │
  ├─► loads embedding from unlabeled/.pt file
  │
  ├─► checks if name already exists in SQLite persons table
  │         │
  │         ├─ YES → _add_embedding_to_person(person_id, embedding)
  │         │           └─ appends to in-memory list + overwrites person_N.pt
  │         │
  │         └─ NO  → _create_person(name, embedding, timestamp)
  │                     └─ inserts SQLite row + saves person_N.pt + creates PersonRecord
  │
  ├─► marks unlabeled row as reviewed = 1 in SQLite
  │
  └─► _add_to_faiss(embedding, person_id)   ← incremental add, no full rebuild
```

### Saving / Loading the Index

```
save_faiss_index()
  ├─► faiss.write_index() → faiss_index.bin   (binary, compressed vectors)
  └─► json.dump()         → faiss_mapping.json (parallel ID list)

load_faiss_index()
  ├─► faiss.read_index()  ← faiss_index.bin
  └─► json.load()         ← faiss_mapping.json
```

---

## FAISS Index Types — Technical Deep Dive

### IndexFlatIP — Exact Brute Force

```
All embeddings stored as-is in memory.
Query vector is compared against every single stored vector.
No compression. No approximation. 100% accurate.

Time complexity: O(n · d)   where n = number of embeddings, d = dimensions (768)
Memory:          O(n · d · 4 bytes) — raw float32 storage
```

**Inner Product (IP):** For L2-normalized vectors, inner product equals cosine similarity.
The score ranges from -1.0 (opposite direction) to 1.0 (identical direction).
The "direction" a vector points = the identity it encodes. Magnitude is irrelevant.

Use when: `n < 5,000`. Still milliseconds at this scale. No training required.

---

### IndexIVFFlat — Approximate with Voronoi Partitioning

At 10,000+ embeddings, brute force becomes noticeable. IVF splits the vector space
into `nlist` regions (cells) using k-means clustering, then only searches nearby cells.

**Step 1 — Training (k-means clustering):**
```
All existing embeddings → k-means → nlist centroids

Each centroid is the "average center" of a cluster of similar faces.
Training is required to place the centroids correctly — IVF must see representative data.
You cannot search an untrained IVFFlat index.
```

**Step 2 — Assigning vectors to cells:**
```
Each embedding gets assigned to its closest centroid.
Stored in an inverted file: centroid_id → [list of embeddings in this cell]
```

**Step 3 — Searching with nprobe:**
```
query → find nprobe nearest centroids (via quantizer, which is an IndexFlatIP)
      → only search those nprobe cells (not all nlist)
      → return the best match found within those cells
```

```
nprobe trade-off:
  nprobe = 1   → fastest, least accurate (may miss the true best match)
  nprobe = 10  → good balance (project default)
  nprobe = nlist → same as brute force, no speedup
```

**The quantizer** is itself an `IndexFlatIP`. It holds only the `nlist` centroids and
does an exact search to find which centroid the query is closest to. This is fast
because `nlist` (100) is tiny compared to the total embedding count.

```
IndexIVFFlat structure:
  quantizer (IndexFlatIP):  [centroid_0, centroid_1, ... centroid_99]  ← 100 vectors
  inverted file:
    cell 0:  [emb_4, emb_11, emb_203, ...]
    cell 1:  [emb_2, emb_9, emb_77, ...]
    ...
```

**Why METRIC_INNER_PRODUCT and not L2?**
L2 distance measures the straight-line gap between vector endpoints.
For face embeddings, what matters is the *angle* (direction), not the endpoint gap.
Two face vectors can be very far apart in L2 but nearly identical in direction.
Inner product (cosine similarity for normalized vectors) captures angle, not distance.

---

## Function Explanations

### `__init__`
Sets up all three systems from scratch. Calls `_load_all()` before `_rebuild_faiss_index()`
because the FAISS index is built from the embeddings loaded into `self.persons`.
If those two were swapped, the FAISS index would always be empty on startup.

### `_init_tables`
Runs `CREATE TABLE IF NOT EXISTS` — safe to call on every startup.
Three tables:
- **persons** — one row per known person
- **sightings** — one row every time a known person is seen (the activity log)
- **unlabeled** — queue of unknown faces waiting to be labeled

### `_load_all`
Rebuilds the in-memory `self.persons` dict from SQLite + disk.
Uses `SELECT id, name, ...` (not `SELECT *`) to make column order explicit —
`SELECT *` would break silently if the schema ever changes column order.

The `isinstance(embeddings, torch.Tensor)` check handles a schema migration edge case:
old `.pt` files may have saved a single 2D tensor `[N, 768]` instead of a list of 1D tensors.
If it's a matrix, it's split into individual rows so the format is always `list[Tensor]`.

### `_rebuild_faiss_index`
Full rebuild from scratch. Called once at startup.
Iterates over every person and every embedding, building two parallel structures simultaneously:
`all_embs` (the vectors) and `all_ids` (the person IDs at matching positions).

`np.stack(all_embs).astype(np.float32)` — FAISS is a C++ library and is strict:
it only accepts contiguous float32 NumPy arrays. `astype` ensures the dtype is correct
even if PyTorch used float16 or float64 internally.

### `_add_to_faiss`
Incremental add — appends one embedding without rebuilding the entire index.
Used after labeling so the new person is immediately searchable.

**Caveat with IVF:** New embeddings are assigned to the existing clusters at add-time.
If the data distribution shifts significantly (many new face types not represented in
the original training), the cluster assignments become inaccurate. Periodically calling
`_rebuild_faiss_index()` retrains the clusters on the full current dataset.

`.reshape(1, -1)` — `index.add()` always expects a 2D array, shape `(n_vectors, dim)`.
A 1D array of shape `(768,)` would raise an error; reshaping to `(1, 768)` tells FAISS
"this is a batch of 1 vector."

### `save_faiss_index` / `load_faiss_index`
`faiss.write_index()` serializes the full index (centroids, assignments, all vectors) to a binary file.
This avoids rebuilding from `.pt` files on every restart, which would cost time at scale.

The JSON mapping must be saved alongside — without it, the loaded FAISS index has correct
vectors but no way to translate results back to `person_id` values.

`load_faiss_index()` returns `bool` so the caller can fall back to `_rebuild_faiss_index()`
if the `.bin` file doesn't exist (e.g. first run).

### `find_match`
Core recognition function. Called once per detected face per frame.

```python
distances, indices = self._faiss_index.search(query_np, k=1)
```

`search()` always returns 2D arrays even for k=1:
- `distances[0][0]` — the similarity score of the best match
- `indices[0][0]`   — the FAISS row number of the best match

`indices[0][0] == -1` means the index was empty at search time (defensive check).

The threshold `0.65` filters out weak matches. A score below 0.65 means the face
is too different from anyone known — it should be queued as unlabeled.

### `find_top_k`
Returns multiple ranked matches instead of just the best one.
`k = min(k, self._faiss_index.ntotal)` prevents asking for more results than vectors exist
(FAISS raises an error if `k > ntotal`).

Useful for a labeling UI: "Did you mean Alice (0.81), Bob (0.74), or Carol (0.68)?"

### `record_sighting`
Writes one row to the `sightings` table and increments `visit_count` in `persons`.
Updates both SQLite and the in-memory `PersonRecord` dict to keep them consistent.
If only SQLite were updated, `self.persons[person_id].visit_count` would be stale
until the next full `_load_all()`.

### `queue_unlabeled`
Saves two files per unknown face:
- `.png` — for a human to look at in the labeling UI
- `.pt`  — the embedding, so FAISS/SQLite can use it immediately after labeling
  without re-running the model

The timestamp + index naming `2026-02-25_14-30-05_2.png` handles multiple faces
appearing at the exact same second.

### `get_unlabeled`
Returns only unreviewed rows (`reviewed = 0`).
Returns raw dicts (not PersonRecord objects) because these aren't known persons yet.

### `label_face`
The human-in-the-loop bridge. Converts an unlabeled face into a known person.

Two paths:
1. **Name already exists** → calls `_add_embedding_to_person` to add another sample
   of a known person (improves future recognition accuracy via multiple embeddings)
2. **New name** → calls `_create_person` to register them for the first time

After either path, calls `_add_to_faiss` so the new embedding is immediately searchable —
no restart needed.

### `_create_person`
`cursor.lastrowid` gets the auto-incremented `person_id` that SQLite just assigned.
`embedding.unsqueeze(0)` wraps the 1D tensor `[768]` into `[1, 768]` so `_load_all()`
can later split it back into a list with the matrix-to-list pattern.

### `_add_embedding_to_person`
`torch.stack(person.embeddings)` converts the full list of 1D tensors back into one
2D matrix `[N, 768]` and overwrites the `.pt` file.
This means the file always stores N embeddings as a single matrix, growing over time.

### `stats`
`type(self._faiss_index).__name__` returns `"IndexFlatIP"` or `"IndexIVFFlat"` as a
human-readable string without importing anything extra. Useful for debugging and
knowing which index mode is currently active.

---

## Tensor & Vector Shape Walkthrough

This section traces exactly what shape the data is at every step — and explains
**why each reshape exists**.

---

### The Core Rule: FAISS Always Expects 2D

FAISS is a batch search library. It's designed to search for many queries at once.
Because of this, every method that takes vectors **requires a 2D array**, never 1D.

```
1D array → shape (768,)       ← a single embedding.  FAISS rejects this.
2D array → shape (1, 768)     ← a "batch of 1".       FAISS accepts this.
2D array → shape (50, 768)    ← a batch of 50 queries. FAISS accepts this.
```

Dimensions read as: `(number_of_vectors, size_of_each_vector)`.

Every reshape in this file exists for one reason: converting a single 1D vector
into a 2D batch-of-one so FAISS stops complaining.

---

### Step 1 — What comes out of DINOv3

```
FaceEmbedder produces a 1D tensor:

  embedding = [0.12, -0.33, 0.87, 0.04, ..., 0.61]
                                                  │
  shape: (768,)   ← 768 numbers in a single row
  dtype: float32
  device: cpu     ← already moved off GPU
  norm:   1.0     ← L2-normalized, so dot product = cosine similarity
```

This is the "raw ingredient." Everything below is about getting it into a shape
that SQLite, PyTorch, NumPy, and FAISS each individually accept.

---

### Step 2 — Storing in the FAISS index (`_rebuild_faiss_index`)

We have a Python list of many 1D tensors (one per stored embedding).
FAISS needs a single 2D matrix.

```
self.persons contains:
  person 7:  [emb_a, emb_b]        each emb is shape (768,)
  person 12: [emb_c]

Step A — collect into a flat Python list:
  all_embs = [emb_a, emb_b, emb_c]    ← list of 3 tensors, each (768,)

Step B — convert each tensor to numpy:
  all_embs = [emb_a.numpy(), emb_b.numpy(), emb_c.numpy()]
             shapes: (768,)  (768,)  (768,)

Step C — np.stack() → combine into one 2D matrix:
  emb_matrix = np.stack(all_embs)

  Before stack:   [(768,), (768,), (768,)]   ← list of 1D arrays
  After stack:    (3, 768)                   ← one 2D matrix
                   │   │
                   │   └── 768 features per vector
                   └────── 3 vectors total

Step D — .astype(np.float32):
  FAISS is written in C++ and only speaks float32.
  PyTorch may internally use float16 or float64.
  This cast guarantees the correct dtype before handing off.

Step E — index.add(emb_matrix):
  FAISS receives (3, 768) ✓
  Internally lays them out as rows 0, 1, 2 in its flat storage.
```

---

### Step 3 — Adding one new embedding (`_add_to_faiss`)

After labeling a face, we add one embedding without rebuilding everything.
The same "must be 2D" rule applies even for a single vector.

```
embedding arrives as shape (768,)   ← 1D

.numpy()              → (768,)      still 1D, just numpy now
.astype(np.float32)   → (768,)      correct dtype
.reshape(1, -1)       → (1, 768)    ← THIS is the key step

  reshape(1, -1) means:
    "give me 1 row"
    "figure out the column count automatically" (-1 = infer from total size)
    768 elements ÷ 1 row = 768 columns → shape (1, 768)

index.add( (1, 768) ) ✓   FAISS accepts it as a batch of 1.
```

The `-1` in `reshape` is a convenience: you don't have to hardcode 768.
`reshape(1, -1)` always means "make it one row, whatever width needed."

---

### Step 4 — Searching (`find_match` and `find_top_k`)

The query embedding goes through the exact same reshape before search.
The *output* of search is also 2D, which is why we index with `[0][0]`.

```
query_emb arrives as shape (768,)   ← 1D tensor from FaceEmbedder

.numpy()              → (768,)
.astype(np.float32)   → (768,)
.reshape(1, -1)       → (1, 768)    batch of 1 query

index.search( (1, 768), k=1 )
  │                         │
  │                         └── k = how many results to return
  │
  └── FAISS internally:
        computes dot product of query against every stored vector
        returns the top-k highest scores

Returns two 2D arrays:
  distances → shape (1, 1)    ← (n_queries, k)
  indices   → shape (1, 1)

  distances[0][0] = 0.81      ← similarity score of best match
  indices[0][0]   = 2         ← FAISS row number of best match
     │  │
     │  └── second [0]: first result within that query (k=1, so only one result)
     └───── first [0]:  first query in the batch (we only sent 1 query)
```

**Why are the outputs 2D?** Because `search()` supports batched queries.
If you sent 5 queries at once, you'd get `distances` of shape `(5, k)` — one row per query.
Since we always send 1 query, we always get shape `(1, k)` and always index `[0]` first.

```
find_top_k with k=5:
  index.search( (1, 768), k=5 )

  distances → (1, 5)   e.g. [[0.91, 0.83, 0.74, 0.68, 0.55]]
  indices   → (1, 5)   e.g. [[2, 0, 5, 1, 3]]

  distances[0] → [0.91, 0.83, 0.74, 0.68, 0.55]   ← the 5 scores
  indices[0]   → [2, 0, 5, 1, 3]                   ← the 5 FAISS row numbers

  zip(distances[0], indices[0]):
    (0.91, 2) → person_id = _faiss_id_to_person_id[2]
    (0.83, 0) → person_id = _faiss_id_to_person_id[0]
    ...
```

---

### Full Shape Journey — One Query

```
DINOv3 output
  embedding: Tensor (768,)   float32, L2-normalized, on CPU
       │
       │  .numpy()
       ▼
  numpy array: (768,)   float32
       │
       │  .reshape(1, -1)
       ▼
  numpy array: (1, 768)  ← "batch of 1" — FAISS can now accept this
       │
       │  index.search(query, k=1)
       ▼
  distances: (1, 1)    e.g. [[0.81]]
  indices:   (1, 1)    e.g. [[2]]
       │
       │  [0][0]  ← unwrap both outer dimensions
       ▼
  best_sim: 0.81   (Python float)
  best_faiss_idx: 2 (Python int)
       │
       │  _faiss_id_to_person_id[2]
       ▼
  person_id: 12   ← real SQLite ID, passed to record_sighting()
```

---

### Shape Cheat Sheet

| Operation | Input shape | Output shape | Why |
|---|---|---|---|
| `emb.numpy()` | `(768,)` tensor | `(768,)` ndarray | PyTorch → NumPy handoff |
| `.astype(np.float32)` | `(768,)` any dtype | `(768,)` float32 | FAISS C++ requires float32 |
| `.reshape(1, -1)` | `(768,)` | `(1, 768)` | FAISS requires 2D input |
| `np.stack([...])` | list of `(768,)` | `(N, 768)` | build 2D matrix for bulk add |
| `index.add(matrix)` | `(N, 768)` | — | stores N vectors as rows 0…N-1 |
| `index.search(q, k)` | `(1, 768)` | `distances (1,k)`, `indices (1,k)` | batch output mirrors batch input |
| `distances[0][0]` | `(1, 1)` | scalar float | unwrap query dim, then k dim |
| `embedding.unsqueeze(0)` | `(768,)` tensor | `(1, 768)` tensor | same as reshape but stays in PyTorch (used for `.pt` file saving) |

---

## Index Auto-Upgrade Summary

```
                 ┌─────────────────────────────────────────┐
                 │         _rebuild_faiss_index()           │
                 │                                          │
  n embeddings   │   n < 5,000           n ≥ 5,000          │
  in database    │       │                   │              │
                 │       ▼                   ▼              │
                 │  IndexFlatIP         IndexIVFFlat        │
                 │  (exact search)      (approx search)     │
                 │  no training         requires .train()   │
                 │  always accurate     faster at scale     │
                 └─────────────────────────────────────────┘
```

The threshold `IVF_THRESHOLD = 5000` and cluster count `IVF_NLIST = 100` are class-level
constants, easy to tune without touching logic. The `nlist = min(IVF_NLIST, n // 10)`
guard ensures you never create more clusters than `total_vectors / 10` — a FAISS rule
of thumb to prevent nearly empty clusters that would waste memory and hurt accuracy.
