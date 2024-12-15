import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70


class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="indices", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.cluster_manager = None
        self.pq_codebooks = {}  # For Product Quantization (PQ)

        # Ensure the index directory exists
        os.makedirs(self.index_path, exist_ok=True)

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            self.load_indices()

    def generate_database(self, size: int) -> None:
        """Generate the database with random vectors and build the index."""
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        vectors = self._normalize_vectors(vectors)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        """Write vectors to a memory-mapped file."""
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode="w+", shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def load_indices(self) -> None:
        """Load indices (centroids, assignments, and PQ codebooks) from files."""
        centroids_path = os.path.join(self.index_path, "ivf_centroids.npy")
        assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")

        if os.path.exists(centroids_path) and os.path.exists(assignments_path):
            centroids = np.load(centroids_path)
            assignments = np.load(assignments_path)

            self.cluster_manager = ClusterManager(num_clusters=len(centroids), dimension=DIMENSION)
            self.cluster_manager.centroids = centroids
            self.cluster_manager.assignments = assignments
        else:
            raise FileNotFoundError("Centroids or assignments files not found.")

        # Load PQ codebooks for each cluster
        self.pq_codebooks = {}
        for cluster_id in np.unique(self.cluster_manager.assignments):
            pq_path = os.path.join(self.index_path, f"pq_cluster_{cluster_id}.npy")
            if os.path.exists(pq_path):
                self.pq_codebooks[cluster_id] = np.load(pq_path)
            else:
                print(f"Warning: PQ codebook for cluster {cluster_id} not found.")

    def _build_index(self) -> None:
        """Build the clustering and PQ indices and save them to disk."""
        vectors = self.get_all_rows()
        vectors = self._normalize_vectors(vectors)

        self.cluster_manager = ClusterManager(
            num_clusters=max(1, min(len(vectors), int(np.sqrt(len(vectors) / 2)))), dimension=DIMENSION
        )
        self.cluster_manager.cluster_vectors(vectors)

        # Save centroids and assignments to disk
        centroids_path = os.path.join(self.index_path, "ivf_centroids.npy")
        assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")
        np.save(centroids_path, self.cluster_manager.kmeans.cluster_centers_)
        np.save(assignments_path, self.cluster_manager.assignments)

        # Save PQ codebooks to disk
        self.pq_codebooks = {}
        for cluster_id in np.unique(self.cluster_manager.assignments):
            cluster_vectors = vectors[self.cluster_manager.assignments == cluster_id]
            codebook = self._train_pq_codebook(cluster_vectors)
            self.pq_codebooks[cluster_id] = codebook
            pq_path = os.path.join(self.index_path, f"pq_cluster_{cluster_id}.npy")
            np.save(pq_path, codebook)

    def retrieve(self, query: np.ndarray, top_k=5) -> List[int]:
        """Retrieve the top K nearest neighbors for a given query."""
        
        # Lazy load indices if not already loaded
        if not self.cluster_manager or not self.cluster_manager.centroids.any():
            self.load_indices()

        query = self._normalize_vectors(np.array([query]))[0]
        valid_clusters = list(self.pq_codebooks.keys())
        if not valid_clusters:
            return []

        cluster_distances = np.linalg.norm(self.cluster_manager.centroids[valid_clusters] - query, axis=1)
        cluster_ids = np.argsort(cluster_distances)[:max(3, top_k // 2)]

        results = []
        seen_indices = set()

        for cluster_id in cluster_ids:
            if cluster_id in self.pq_codebooks:
                cluster_codebook = self.pq_codebooks[cluster_id]
                pq_results = self._pq_search(cluster_codebook, query, top_k)
                for idx, similarity in pq_results:
                    if idx not in seen_indices:
                        results.append((similarity, idx))
                        seen_indices.add(idx)

        results.sort(reverse=True, key=lambda x: x[0])
        return [idx for _, idx in results[:top_k]]

    def _pq_search(self, codebook: np.ndarray, query: np.ndarray, top_k: int) -> List[tuple]:
        """Search within a PQ codebook for the nearest vectors."""
        quantized_query = self._quantize(codebook, query)
        distances = np.linalg.norm(codebook - quantized_query, axis=1)
        closest_indices = np.argsort(distances)[:top_k]
        return [(idx, distances[idx]) for idx in closest_indices]

    def _quantize(self, codebook: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Quantize a vector to the nearest codebook entry."""
        return codebook[np.argmin(np.linalg.norm(codebook - vector, axis=1))]

    def _train_pq_codebook(self, cluster_vectors: np.ndarray) -> np.ndarray:
        """Train a PQ codebook for a cluster."""
        num_clusters = min(len(cluster_vectors), 256)
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=DB_SEED_NUMBER)
        kmeans.fit(cluster_vectors)
        return kmeans.cluster_centers_

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length."""
        if vectors.ndim == 3 and vectors.shape[0] == 1:
            # Unwrap from (1, 1, 70) to (1, 70)
            vectors = vectors[0]
        elif vectors.ndim == 3:
            # For general 3D arrays, flatten the first dimension
            vectors = vectors.reshape(-1, vectors.shape[-1])

        if vectors.ndim == 1:  # Single vector (70,)
            norms = np.linalg.norm(vectors)
            if norms == 0:
                return vectors
            return vectors / norms
        elif vectors.ndim == 2:  # Batch of vectors (N, 70)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        else:
            raise ValueError(f"Input array must be 1D or 2D. Current shape: {vectors.shape}")


    def get_all_rows(self) -> np.ndarray:
        """Get all rows stored in the database file."""
        num_records = os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
        return np.memmap(self.db_path, dtype=np.float32, mode="r", shape=(num_records, DIMENSION))
    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = int(row_num * DIMENSION * ELEMENT_SIZE)
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve row {row_num}: {e}")

class ClusterManager:
    def __init__(self, num_clusters: int, dimension: int):
        self.num_clusters = num_clusters
        self.dimension = dimension
        self.kmeans = None
        self.centroids = None
        self.assignments = None

    def cluster_vectors(self, vectors: np.ndarray) -> None:
        """Cluster vectors using MiniBatchKMeans."""
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=DB_SEED_NUMBER, batch_size=1024)
        self.assignments = self.kmeans.fit_predict(vectors)
        self.centroids = self.kmeans.cluster_centers_
