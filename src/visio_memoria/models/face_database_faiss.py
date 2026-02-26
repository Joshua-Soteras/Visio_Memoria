"""
face_database_faiss.py — FAISS-Powered Face Database
Drop-in replacement for face_database.py

WHY FAISS?
──────────
Your current approach:  embeddings @ query  (brute-force dot product)
  - Works fine for ~50 people / ~200 embeddings
  - Gets slow at 1000+ embeddings (linear scan every frame)

FAISS:
  - IndexFlatIP   → exact same results, but optimized C++ (2-5x faster)
  - IndexIVFFlat  → approximate search, 10-50x faster at 10k+ embeddings
  - IndexHNSW     → graph-based, great for 1k-100k scale

This file uses IndexFlatIP for small databases and auto-upgrades
to IndexIVFFlat when you cross a configurable threshold.

INSTALL:
  pip install faiss-cpu          # CPU-only (fine for this project)
  # or: pip install faiss-gpu    # if you have CUDA

USAGE (same API as face_database.py):
  from face_database_faiss import FaceDatabaseFAISS
  db = FaceDatabaseFAISS()
  person_id, similarity = db.find_match(embedding)
"""

import torch
import numpy as np
import faiss
import sqlite3
from pathlib import Path
from datetime import datetime
from PIL import Image
from dataclasses import dataclass, field


@dataclass
class PersonRecord:
    """A known person in the database."""
    person_id: int
    name: str
    embeddings: list[torch.Tensor] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    visit_count: int = 0
    notes: str = ""


class FaceDatabaseFAISS:
    """
    FAISS-backed face database.

    How it works:
    ─────────────
    FAISS stores all embeddings in a flat array internally.
    We maintain a parallel list `_faiss_id_to_person_id` that maps
    each FAISS index position → person_id in SQLite.

    When you call find_match(embedding):
      1. FAISS searches its index → returns (distance, faiss_index)
      2. We map faiss_index → person_id via our lookup list
      3. Return (person_id, similarity)

    Index types:
    ─────────────
    - IndexFlatIP:  Exact inner product (cosine sim for L2-normed vectors)
                    Best for < 5,000 embeddings. No training needed.

    - IndexIVFFlat: Inverted file index. Partitions vectors into clusters,
                    only searches nearby clusters. ~10-50x faster for 10k+.
                    Requires training on existing vectors first.

    Storage layout (same as before):
        db_path/
        ├── lumemoria.db
        ├── faiss_index.bin        # ← NEW: serialized FAISS index
        ├── faiss_mapping.json     # ← NEW: faiss_idx → person_id map
        ├── embeddings/
        └── unlabeled/
    """

    # Auto-upgrade to IVF when embedding count exceeds this
    IVF_THRESHOLD = 5000
    # Number of clusters for IVF (rule of thumb: sqrt(n))
    IVF_NLIST = 100
    # Number of clusters to search (higher = more accurate, slower)
    IVF_NPROBE = 10

    def __init__(self, db_path: str = "", embed_dim: int = 768):
        self.db_path = Path(db_path)
        self.embeddings_dir = self.db_path / "embeddings"
        self.unlabeled_dir = self.db_path / "unlabeled"
        self.embed_dim = embed_dim

        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.unlabeled_dir.mkdir(parents=True, exist_ok=True)

        # SQLite
        #conn = connect to the database 
        self.conn = sqlite3.connect(str(self.db_path / "visio_memoria.db"))
        self._init_tables()

        # Person records (same as before)
        self.persons: dict[int, PersonRecord] = {}
        self._load_all()

        # FAISS index
        self._faiss_index: faiss.Index | None = None
        self._faiss_id_to_person_id: list[int] = []  # faiss row → person_id
        self._rebuild_faiss_index()


    #=========================================
    # SQLITE (unchanged)
    #=========================================
    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                first_seen TEXT,
                last_seen TEXT,
                visit_count INTEGER DEFAULT 0,
                notes TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER REFERENCES persons(person_id),
                timestamp TEXT,
                similarity REAL,
                notes TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS unlabeled (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                embedding_path TEXT,
                timestamp TEXT,
                reviewed INTEGER DEFAULT 0
            );
        """)
        self.conn.commit()



    #=========================================
    # LOADING (unchanged)
    #=========================================
    def _load_all(self):

        #1select all the cols FROM table name 
        #self.conn... returns a "cursor" object
        #cursor is just hovering over the results 
        # Safer alternative than the original code :
        cursor = self.conn.execute("SELECT id, name, first_seen, last_seen, visit_count, notes FROM persons")
        #cursor = self.conn.execute("SELECT * FROM persons")

        #2 fetching rows one by one 
        #  each row is a tuple
        for row in cursor:
            
            #3 extracting information from tuple
            pid, name, first_seen, last_seen, visit_count, notes = row

            #4 create a file path for this specific embedding
            emb_path = self.embeddings_dir / f"person_{pid}.pt"

            #5 create embeddings list 
            embeddings = []

            #6 checking if the file exist 
            if emb_path.exists():

                #load all embeddings on the device 
                #weights_only = True great security feature to prevent loading malicious code
                embeddings = torch.load(emb_path, weights_only=True)
                
                #if if the tensor obtained does not match our list[torch.tensor]
                #example [5,128 ] -> 5 seperate 1D (128) vectors. we combine them to make matrix 
                if isinstance(embeddings, torch.Tensor):
                    embeddings = [embeddings[i] for i in range(embeddings.shape[0])]

            self.persons[pid] = PersonRecord(
                person_id=pid, name=name, embeddings=embeddings,
                first_seen=first_seen, last_seen=last_seen,
                visit_count=visit_count, notes=notes,
            )
        print(f"Loaded {len(self.persons)} known persons")



    #=========================================
    # FAISS INDEX MANAGEMENT
    #=========================================
    def _rebuild_faiss_index(self):
        """
        Build (or rebuild) the FAISS index from all stored embeddings.

        Strategy:
        - < IVF_THRESHOLD embeddings → IndexFlatIP (exact, no training)
        - ≥ IVF_THRESHOLD embeddings → IndexIVFFlat (approximate, needs training)


        IndexFlatIP 
            - flat means brute force search 
                - FAISS will compare the search query against every single
                  uncompressed vector in the database to find the 100% perfect match
            - IP = inner product 
                - measures to see if the angle between the vectors 
                - sees if they are pointing to the same directon 
        """

        all_embs = []
        all_ids = []

        for pid, person in self.persons.items():
            for emb in person.embeddings:

                #converts the pytorch tensors to standard numpy arrays 
                all_embs.append(emb.numpy())

                #add person's id to the parallel list
                #ensures the id is at the same exact index postion 
                all_ids.append(pid)

        # save the paralledl id list to the class so other methods can access it 
        self._faiss_id_to_person_id = all_ids

        #get total number of embeddings 
        n = len(all_embs)

        if n == 0:
            # Empty index — use flat so we can add to it without training
            self._faiss_index = faiss.IndexFlatIP(self.embed_dim)
            print("FAISS index: empty (IndexFlatIP)")
            return
        
        # Stack into 2D numpy matrix (
        # .astype ensures FAISS needs float32 contiguous) 
        emb_matrix = np.stack(all_embs).astype(np.float32)

        #determining which index we should build
        if n < self.IVF_THRESHOLD:
            # ── EXACT SEARCH (IndexFlatIP) ──
            #brute force will still be fast enough at this point 
            # Inner product on  vectors = cosine similarity
            self._faiss_index = faiss.IndexFlatIP(self.embed_dim)
            self._faiss_index.add(emb_matrix)
            print(f"FAISS index: IndexFlatIP with {n} embeddings")

        else:
            # ── APPROXIMATE SEARCH (IndexIVFFlat) ──
            #the database is getting massive
            #create Voroni cells and centroids
            # Partition space into clusters, search only nearby ones

            #Calculate how many clusters to make 
            # don't create more clusters than total_data / 10
            nlist = min(self.IVF_NLIST, n // 10)  # don't have more clusters than data/10

            #coarse quantizer = the measuring tape for the clusters
            #find the centroid cluster that is closest to the query vector 
            #quantizer only holds the centroids 
            #self.embed_dim is the dimensions of the vectors 
            quantizer = faiss.IndexFlatIP(self.embed_dim)


            #Why choose Metric_innter_product
            #n the world of embeddings, the direction a vector points represents its semantic meaning,
            # while its magnitude (length) often just represents the frequency of words 
            # or the confidence of the model.
            self._faiss_index = faiss.IndexIVFFlat(
                quantizer, self.embed_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )


            # IVF requires training on representative data
            # needs to know where to place the cluster centroids 
            self._faiss_index.train(emb_matrix)

            #after train, the centroids are laid out, push the data to clusters 
            self._faiss_index.add(emb_matrix)

            # Set how many clusters FAISS should check during a search
            # (higher = more accurate, but slower)
            self._faiss_index.nprobe = self.IVF_NPROBE
            print(f"FAISS index: IndexIVFFlat with {n} embeddings, {nlist} clusters")


    def _add_to_faiss(self, embedding: torch.Tensor, person_id: int):
        """
        Add a single embedding to the FAISS index without rebuilding.
        
        NOTE: If we're using IVF and the data distribution shifts significantly,
        you should call _rebuild_faiss_index() periodically to retrain clusters.
        """
        emb_np = embedding.numpy().astype(np.float32).reshape(1, -1)
        self._faiss_index.add(emb_np)
        self._faiss_id_to_person_id.append(person_id)


    def save_faiss_index(self):
        
        """Persist the FAISS index to disk for faster restarts."""
        if self._faiss_index is not None:
            #save the actual FAISS math/vectprs into highly compressed binary file (.bin)
            index_path = str(self.db_path / "faiss_index.bin")
            faiss.write_index(self._faiss_index, index_path)

            # 2. Save our parallel Python list of IDs into a standard JSON file.
            # We MUST save both, otherwise FAISS will find matches but we won't know who they belong to!
            import json
            mapping_path = str(self.db_path / "faiss_mapping.json")
            with open(mapping_path, "w") as f:
                json.dump(self._faiss_id_to_person_id, f)

            print(f"Saved FAISS index ({self._faiss_index.ntotal} vectors)")


    def load_faiss_index(self) -> bool:

        """Load a previously saved FAISS index. Returns True if successful."""
        import json
        index_path = self.db_path / "faiss_index.bin"
        mapping_path = self.db_path / "faiss_mapping.json"

        if index_path.exists() and mapping_path.exists():

            #loading the vector math back into FAISS memory 
            self._faiss_index = faiss.read_index(str(index_path))
            with open(mapping_path) as f:
                self._faiss_id_to_person_id = json.load(f)
            print(f"Loaded FAISS index ({self._faiss_index.ntotal} vectors)")
            return True
        
        return False



    #=========================================
    # MATCHING (FAISS-powered)
    #=========================================
    def find_match(self, query_emb: torch.Tensor, threshold: float = 0.65) -> tuple[int | None, float]:
        """
        Find the best matching person using FAISS.

        How FAISS search works:
        ───────────────────────
        index.search(query, k) returns:
          - distances: shape (1, k) — similarity scores
          - indices:   shape (1, k) — positions in the index

        For IndexFlatIP with L2-normalized vectors:
          distance = cosine similarity (1.0 = identical, 0.0 = orthogonal)

        Args:
            query_emb: L2-normalized embedding, shape (embed_dim,)
            threshold: Minimum cosine similarity to count as match

        Returns:
            (person_id, similarity) or (None, best_similarity)
        """

        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return None, 0.0

        # FAISS expects float32 numpy, shape (1, dim)
        #pytorch 1d tensor and reshapes it 
        query_np = query_emb.numpy().astype(np.float32).reshape(1, -1)

        # Search for top-1 match
        #returns back two list
        #distances  (similarity scores )
        # indicies (the row numbers)
        #self._faiss_index .search method is determined above (cosine similary vs euclidean distance)
        distances, indices = self._faiss_index.search(query_np, k=1)

        #extract the actual numbers from those lists 
        best_sim = float(distances[0][0])
        best_faiss_idx = int(indices[0][0])

        # FAISS returns -1 for empty results
        if best_faiss_idx < 0:
            return None, 0.0

        #if the similarity score is higher than 65 person 
        # 
        if best_sim >= threshold:

            #look in parallet python list to get the real person id
            person_id = self._faiss_id_to_person_id[best_faiss_idx]
            return person_id, best_sim

        #score is too low return nothing
        return None, best_sim


    def find_top_k(self, query_emb: torch.Tensor, k: int = 5) -> list[tuple[int, float]]:
        """
        Find the top-k most similar persons.
        Useful for debugging or showing "did you mean?" suggestions.

        Returns:
            List of (person_id, similarity) sorted by similarity desc
        """
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        #make sure we don't ask for 5 results if the data
        k = min(k, self._faiss_index.ntotal)
        query_np = query_emb.numpy().astype(np.float32).reshape(1, -1)
        distances, indices = self._faiss_index.search(query_np, k=k)


        #loop through the results, grab the real person ids, and put then in a list 
        results = []
        for sim, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            pid = self._faiss_id_to_person_id[int(idx)]
            results.append((pid, float(sim)))
        return results



    #=========================================
    # RECOGNITION EVENT (unchanged)
    # updating the database once we see someone 
    #=========================================
    def record_sighting(self, person_id: int, similarity: float):
        now = datetime.now().isoformat()


        self.conn.execute(
            "INSERT INTO sightings (person_id, timestamp, similarity) VALUES (?, ?, ?)",
            (person_id, now, similarity)
        )
        self.conn.execute(
            "UPDATE persons SET last_seen = ?, visit_count = visit_count + 1 WHERE person_id = ?",
            (now, person_id)
        )

        #keepiing the python dictionary synced with the database 
        self.conn.commit()
        person = self.persons[person_id]
        person.last_seen = now
        person.visit_count += 1



    #=========================================
    # UNLABELED QUEUE (unchanged)
    # adding strangers to a queue to be identitfied 
    #=========================================
    def queue_unlabeled(self, face_image: Image.Image, embedding: torch.Tensor) -> str:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        existing = list(self.unlabeled_dir.glob(f"{timestamp}_*.png"))
        idx = len(existing)

        #save the picture of the face , human can looj at it later 
        img_path = self.unlabeled_dir / f"{timestamp}_{idx}.png"

        #save the math/vector (.pt) so the ai can use it later 
        emb_path = self.unlabeled_dir / f"{timestamp}_{idx}.pt"
        face_image.save(str(img_path))
        torch.save(embedding, str(emb_path))


        #add this person to the unlabeled database
        self.conn.execute(
            "INSERT INTO unlabeled (image_path, embedding_path, timestamp) VALUES (?, ?, ?)",
            (str(img_path), str(emb_path), now.isoformat())
        )
        self.conn.commit()
        print(f"  📸 Queued unknown face: {img_path.name}")
        return str(img_path)



    #=========================================
    # LABELING (updated to maintain FAISS index)
    #=========================================
    def get_unlabeled(self) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT id, image_path, embedding_path, timestamp FROM unlabeled WHERE reviewed = 0"
        )
        return [
            {"id": r[0], "image_path": r[1], "embedding_path": r[2], "timestamp": r[3]}
            for r in cursor
        ]


    def label_face(self, unlabeled_id: int, name: str) -> int:

        #.fetchone() grabs one specific row instead of returning a list
        row = self.conn.execute(
            "SELECT embedding_path, timestamp FROM unlabeled WHERE id = ?", (unlabeled_id,)
        ).fetchone()

        if not row:
            raise ValueError(f"Unlabeled record {unlabeled_id} not found")

        #load their math/vector off the hard dribve 
        emb_path, timestamp = row
        embedding = torch.load(emb_path, weights_only=True)

        #check if the name the human typed already exists with the system
        existing = self.conn.execute(
            "SELECT person_id FROM persons WHERE name = ?", (name,)
        ).fetchone()

        #if the person exists
        if existing:
            person_id = existing[0]
            self._add_embedding_to_person(person_id, embedding)
        else:
            person_id = self._create_person(name, embedding, timestamp)

        self.conn.execute("UPDATE unlabeled SET reviewed = 1 WHERE id = ?", (unlabeled_id,))
        self.conn.commit()

        # Add to FAISS incrementally (no full rebuild needed)
        self._add_to_faiss(embedding, person_id)

        print(f"  ✅ Labeled as '{name}' (person_id={person_id})")
        return person_id


    def _create_person(self, name: str, embedding: torch.Tensor, first_seen: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO persons (name, first_seen, last_seen, visit_count) VALUES (?, ?, ?, 1)",
            (name, first_seen, first_seen)
        )
        person_id = cursor.lastrowid
        self.conn.commit()

        emb_path = self.embeddings_dir / f"person_{person_id}.pt"
        torch.save(embedding.unsqueeze(0), str(emb_path))

        self.persons[person_id] = PersonRecord(
            person_id=person_id, name=name, embeddings=[embedding],
            first_seen=first_seen, last_seen=first_seen, visit_count=1,
        )
        return person_id


    def _add_embedding_to_person(self, person_id: int, embedding: torch.Tensor):
        person = self.persons[person_id]
        person.embeddings.append(embedding)

        emb_path = self.embeddings_dir / f"person_{person_id}.pt"
        torch.save(torch.stack(person.embeddings), str(emb_path))



    # ─────────────────────────────────────────
    # ACTIVITY NOTES (unchanged)
    # ─────────────────────────────────────────
    def update_notes(self, person_id: int, notes: str):
        self.conn.execute(
            "UPDATE persons SET notes = ? WHERE person_id = ?", (notes, person_id)
        )
        self.conn.commit()
        self.persons[person_id].notes = notes

    # ─────────────────────────────────────────
    # INFO (unchanged)
    # ─────────────────────────────────────────

    def get_person(self, person_id: int) -> PersonRecord | None:
        return self.persons.get(person_id)

    def get_all_persons(self) -> list[PersonRecord]:
        return list(self.persons.values())

    def stats(self) -> dict:
        unlabeled_count = self.conn.execute(
            "SELECT COUNT(*) FROM unlabeled WHERE reviewed = 0"
        ).fetchone()[0]

        index_type = "none"
        index_size = 0
        if self._faiss_index is not None:
            index_type = type(self._faiss_index).__name__
            index_size = self._faiss_index.ntotal

        return {
            "known_persons": len(self.persons),
            "total_embeddings": sum(len(p.embeddings) for p in self.persons.values()),
            "unlabeled_queue": unlabeled_count,
            "faiss_index_type": index_type,
            "faiss_index_size": index_size,
        }


# ─────────────────────────────────────────────
# BENCHMARK: Compare flat torch vs FAISS
# ─────────────────────────────────────────────

def benchmark():
    """
    Demonstrates the speed difference between flat torch search
    and FAISS at different scales.
    """
    import time

    dim = 768  # DINOv2 ViT-B/14

    for n in [100, 1_000, 10_000, 50_000]:
        print(f"\n{'='*50}")
        print(f"  Benchmark: {n:,} embeddings, dim={dim}")
        print(f"{'='*50}")

        # Generate random normalized embeddings
        db_embs = torch.randn(n, dim)
        db_embs = db_embs / db_embs.norm(dim=1, keepdim=True)
        query = torch.randn(dim)
        query = query / query.norm()

        # ── Method 1: Flat torch (your current approach) ──
        times = []
        for _ in range(100):
            start = time.perf_counter()
            sims = db_embs @ query
            best_idx = sims.argmax().item()
            best_sim = sims[best_idx].item()
            times.append(time.perf_counter() - start)
        torch_ms = np.mean(times) * 1000
        print(f"  Torch flat:    {torch_ms:.3f} ms/query")

        # ── Method 2: FAISS IndexFlatIP ──
        db_np = db_embs.numpy().astype(np.float32)
        q_np = query.numpy().astype(np.float32).reshape(1, -1)

        index_flat = faiss.IndexFlatIP(dim)
        index_flat.add(db_np)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            D, I = index_flat.search(q_np, 1)
            times.append(time.perf_counter() - start)
        faiss_flat_ms = np.mean(times) * 1000
        print(f"  FAISS Flat:    {faiss_flat_ms:.3f} ms/query  ({torch_ms/faiss_flat_ms:.1f}x faster)")

        # ── Method 3: FAISS IndexIVFFlat (only for larger sets) ──
        if n >= 1000:
            nlist = min(100, n // 10)
            quantizer = faiss.IndexFlatIP(dim)
            index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index_ivf.train(db_np)
            index_ivf.add(db_np)
            index_ivf.nprobe = 10

            times = []
            for _ in range(100):
                start = time.perf_counter()
                D, I = index_ivf.search(q_np, 1)
                times.append(time.perf_counter() - start)
            faiss_ivf_ms = np.mean(times) * 1000
            print(f"  FAISS IVF:     {faiss_ivf_ms:.3f} ms/query  ({torch_ms/faiss_ivf_ms:.1f}x faster)")

        # Verify same result
        D_flat, I_flat = index_flat.search(q_np, 1)
        sims = db_embs @ query
        torch_best = sims.argmax().item()
        print(f"  Results match: {I_flat[0][0] == torch_best}")


if __name__ == "__main__":
    benchmark()
