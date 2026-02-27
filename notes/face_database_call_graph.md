# FaceDatabaseFAISS — Function Call Graph

How every function connects to every other function.
Arrows mean "calls" — left/top calls right/bottom.

---

## Legend

```
[PublicMethod]    ← called from outside the class (entry points)
[_privateMethod]  ← only called by other methods inside the class
[FAISS / SQLite]  ← external library calls
```

---

## Master Call Graph — All Relationships

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                                 │
│   (these are the only functions called from outside the class)      │
├──────────────┬────────────┬─────────────┬──────────────┬────────────┤
│  __init__    │ find_match │ label_face  │queue_unlabeled│save/load  │
│              │ find_top_k │             │               │_faiss_    │
│              │            │             │               │ index     │
└──────┬───────┴─────┬──────┴──────┬──────┴───────────────┴─────┬─────┘
       │             │             │                             │
       ▼             │             │                             │
┌──────────────┐     │             │                             │
│_init_tables  │     │             │                             │
│              │     │             │                             │
│  SQLite:     │     │             │                             │
│  CREATE TABLE│     │             │                             │
└──────────────┘     │             │                             │
       │             │             │                             │
       ▼             │             │                             │
┌──────────────┐     │             │                             │
│  _load_all   │     │             │                             │
│              │     │             │                             │
│  SQLite:     │     │             │                             │
│  SELECT FROM │     │             │                             │
│  persons     │     │             │                             │
│              │     │             │                             │
│  torch.load  │     │             │                             │
│  person_N.pt │     │             │                             │
└──────────────┘     │             │                             │
       │             │             │                             │
       ▼             │             │                             │
┌──────────────────┐ │             │                             │
│_rebuild_faiss_   │ │             │                             │
│    index         │ │             │                             │
│                  │ │             │                             │
│  reads:          │ │             │                             │
│  self.persons    │ │             │                             │
│                  │ │             │                             │
│  builds either:  │ │             │                             │
│  faiss.Index     │ │             │                             │
│  FlatIP  ──OR──  │ │             │                             │
│  faiss.Index     │ │             │                             │
│  IVFFlat         │ │             │                             │
└──────────────────┘ │             │                             │
                     │             │                             │
       ┌─────────────┘             │                             │
       ▼                           │                             │
┌──────────────────┐               │                             │
│  _faiss_index    │               │                             │
│  .search()       │               │                             │
│                  │               │                             │
│  returns:        │               │                             │
│  distances (1,k) │               │                             │
│  indices   (1,k) │               │                             │
└──────────────────┘               │                             │
                                   │                             │
       ┌───────────────────────────┘                             │
       ▼                                                         │
┌─────────────────────────────────────────┐                      │
│              label_face                 │                      │
│                                         │                      │
│  SQLite: SELECT FROM persons            │                      │
│  (check if name already exists)         │                      │
│                                         │                      │
│       ┌─────────────┴─────────────┐     │                      │
│       ▼                           ▼     │                      │
│  name exists?               new name?  │                      │
│       │                           │     │                      │
│       ▼                           ▼     │                      │
│ _add_embedding_           _create_     │                      │
│  _to_person               person       │                      │
│       │                       │         │                      │
│       └──────────┬────────────┘         │                      │
│                  ▼                      │                      │
│          _add_to_faiss                  │                      │
└─────────────────────────────────────────┘                      │
                                                                 │
       ┌─────────────────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────┐
│         save_faiss_index                 │
│           faiss.write_index → .bin       │
│           json.dump         → .json      │
│                                          │
│         load_faiss_index                 │
│           faiss.read_index  ← .bin       │
│           json.load         ← .json      │
└──────────────────────────────────────────┘
```

---

## Scenario 1 — Startup (`__init__`)

Every function that fires when `FaceDatabaseFAISS()` is created.

```
FaceDatabaseFAISS()
  │
  ├─1─► _init_tables()
  │       │
  │       └── SQLite.executescript()
  │             CREATE TABLE IF NOT EXISTS persons
  │             CREATE TABLE IF NOT EXISTS sightings
  │             CREATE TABLE IF NOT EXISTS unlabeled
  │
  ├─2─► _load_all()
  │       │
  │       ├── SQLite.execute("SELECT ... FROM persons")
  │       │
  │       └── for each person row:
  │               torch.load("embeddings/person_N.pt")
  │               → self.persons[pid] = PersonRecord(...)
  │
  └─3─► _rebuild_faiss_index()
          │
          ├── reads self.persons  (populated by _load_all above)
          │
          ├── np.stack(all_embs)  → emb_matrix (N, 768)
          │
          └── if n < 5000:
          │     faiss.IndexFlatIP(768)
          │     index.add(emb_matrix)
          │
          └── if n ≥ 5000:
                quantizer = faiss.IndexFlatIP(768)
                faiss.IndexIVFFlat(quantizer, ...)
                index.train(emb_matrix)
                index.add(emb_matrix)
                index.nprobe = 10
```

---

## Scenario 2 — Recognition (`find_match`)

Called once per detected face, every frame.

```
find_match(query_emb, threshold=0.65)
  │
  ├── guard: is index empty? → return (None, 0.0)
  │
  ├── query_emb.numpy().reshape(1, -1)       shape: (768,) → (1, 768)
  │
  ├── self._faiss_index.search(query_np, k=1)
  │     │
  │     └── returns distances (1,1), indices (1,1)
  │
  ├── best_sim = distances[0][0]             unwrap to scalar float
  ├── best_faiss_idx = indices[0][0]         unwrap to scalar int
  │
  ├── guard: best_faiss_idx < 0? → return (None, 0.0)
  │
  ├── best_sim >= threshold?
  │     │
  │     ├── YES → person_id = _faiss_id_to_person_id[best_faiss_idx]
  │     │         return (person_id, best_sim)
  │     │
  │     └── NO  → return (None, best_sim)
  │
  └── [caller typically calls record_sighting() with the returned person_id]
```

```
find_top_k(query_emb, k=5)        ← same shape pipeline, k results instead of 1
  │
  ├── k = min(k, index.ntotal)     guard: don't ask for more than exists
  ├── query_emb.numpy().reshape(1, -1)
  ├── self._faiss_index.search(query_np, k=k)
  │     └── returns distances (1,k), indices (1,k)
  │
  └── for sim, idx in zip(distances[0], indices[0]):
          pid = _faiss_id_to_person_id[idx]
          results.append( (pid, sim) )
```

---

## Scenario 3 — Unknown Face (`queue_unlabeled`)

Called when `find_match` returns `None`.

```
queue_unlabeled(face_image, embedding)
  │
  ├── builds file paths:
  │     unlabeled/2026-02-25_14-30-05_0.png
  │     unlabeled/2026-02-25_14-30-05_0.pt
  │
  ├── face_image.save(img_path)          save PNG for human review
  ├── torch.save(embedding, emb_path)    save tensor for AI reuse
  │
  └── SQLite.execute(INSERT INTO unlabeled ...)
```

---

## Scenario 4 — Labeling a Face (`label_face`)

Called when a human identifies an unknown face in the UI.

```
label_face(unlabeled_id, name)
  │
  ├── SQLite.execute("SELECT embedding_path FROM unlabeled WHERE id=?")
  │     └── .fetchone()
  │
  ├── torch.load(emb_path)             load the stored embedding off disk
  │
  ├── SQLite.execute("SELECT person_id FROM persons WHERE name=?")
  │     └── .fetchone()
  │           │
  │           ├── row found (person exists):
  │           │       └─► _add_embedding_to_person(person_id, embedding)
  │           │                 │
  │           │                 ├── person.embeddings.append(embedding)
  │           │                 └── torch.save( torch.stack(embeddings), person_N.pt )
  │           │                       overwrites file with updated matrix (N+1, 768)
  │           │
  │           └── row not found (new person):
  │                   └─► _create_person(name, embedding, timestamp)
  │                             │
  │                             ├── SQLite.execute(INSERT INTO persons ...)
  │                             ├── cursor.lastrowid → new person_id
  │                             ├── torch.save(embedding.unsqueeze(0), person_N.pt)
  │                             └── self.persons[person_id] = PersonRecord(...)
  │
  ├── SQLite.execute("UPDATE unlabeled SET reviewed=1 WHERE id=?")
  │
  └─► _add_to_faiss(embedding, person_id)
            │
            ├── embedding.numpy().reshape(1, -1)   shape: (1, 768)
            ├── self._faiss_index.add( (1, 768) )  append to FAISS array
            └── _faiss_id_to_person_id.append(person_id)
```

---

## Scenario 5 — Save & Load Index

```
save_faiss_index()
  │
  ├── faiss.write_index(self._faiss_index, "faiss_index.bin")
  │     serializes entire index (all vectors + structure) to binary
  │
  └── json.dump(_faiss_id_to_person_id, "faiss_mapping.json")
        serializes parallel ID list as plain JSON


load_faiss_index() → bool
  │
  ├── check: faiss_index.bin exists AND faiss_mapping.json exists?
  │     │
  │     ├── NO  → return False
  │     │           caller falls back to _rebuild_faiss_index()
  │     │
  │     └── YES →
  │             self._faiss_index = faiss.read_index("faiss_index.bin")
  │             self._faiss_id_to_person_id = json.load("faiss_mapping.json")
  │             return True
```

---

## Who Calls Who — Quick Reference Table

| Function | Called by | Calls |
|---|---|---|
| `__init__` | external | `_init_tables`, `_load_all`, `_rebuild_faiss_index` |
| `_init_tables` | `__init__` | SQLite |
| `_load_all` | `__init__` | SQLite, `torch.load` |
| `_rebuild_faiss_index` | `__init__` (also: manual rebuild) | `faiss.IndexFlatIP`, `faiss.IndexIVFFlat` |
| `_add_to_faiss` | `label_face` | `self._faiss_index.add` |
| `save_faiss_index` | external | `faiss.write_index`, `json.dump` |
| `load_faiss_index` | external | `faiss.read_index`, `json.load` |
| `find_match` | external | `self._faiss_index.search` |
| `find_top_k` | external | `self._faiss_index.search` |
| `record_sighting` | external | SQLite |
| `queue_unlabeled` | external | `face_image.save`, `torch.save`, SQLite |
| `get_unlabeled` | external | SQLite |
| `label_face` | external | `_add_embedding_to_person` OR `_create_person`, then `_add_to_faiss`, SQLite |
| `_create_person` | `label_face` | SQLite, `torch.save` |
| `_add_embedding_to_person` | `label_face` | `torch.save` |
| `update_notes` | external | SQLite |
| `get_person` | external | `self.persons` dict |
| `get_all_persons` | external | `self.persons` dict |
| `stats` | external | SQLite, `self._faiss_index` |
