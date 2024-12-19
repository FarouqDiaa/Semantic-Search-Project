import os
import numpy as np
from typing import List

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db_100.dat", index_file_path="saved_db_100", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.cluster_manager = None

        os.makedirs(self.index_path, exist_ok=True)

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = np.memmap(self.db_path, dtype=np.float32, mode="w+", shape=(size, DIMENSION))
        for start in range(0, size, 100000):
            end = min(start + 100000, size)
            vectors[start:end] = rng.random((end - start, DIMENSION), dtype=np.float32)
        vectors.flush()
        self._build_index(full_rebuild=True)

    def load_indices(self) -> None:
        centroids_path = os.path.join(self.index_path, "ivf_centroids.npy")
        assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")

        if os.path.exists(centroids_path) and os.path.exists(assignments_path):
            self.cluster_manager = ClusterManager(num_clusters=None, dimension=DIMENSION)
            self.cluster_manager.centroids = np.load(centroids_path)
            self.cluster_manager.assignments = np.load(assignments_path)
        else:
            raise FileNotFoundError("Centroids or assignments files not found.")

    def _build_index(self, full_rebuild=False):
        vectors = self.get_all_rows()

        if full_rebuild:
            num_clusters = max(1, min(len(vectors), int(np.sqrt(len(vectors) / 2))))
            self.cluster_manager = ClusterManager(num_clusters, dimension=DIMENSION)
            self.cluster_manager.cluster_vectors(vectors)

            # Save centroids and assignments to disk
            np.save(os.path.join(self.index_path, "ivf_centroids.npy"), self.cluster_manager.centroids)
            np.save(os.path.join(self.index_path, "ivf_assignments.npy"), self.cluster_manager.assignments)

    def retrieve(self, query: np.ndarray, top_k: int) -> List[int]:
        if self.cluster_manager is None:
            self.load_indices()
    
        # Step 1: Calculate distances to centroids (vectorized operation)
        centroid_distances = np.linalg.norm(self.cluster_manager.centroids - query, axis=1)
        sorted_centroid_indices = np.argsort(centroid_distances)
    
        # Step 2: Select top clusters to search
        max_clusters_to_search = max(5, min(len(sorted_centroid_indices), top_k * 8))
        top_cluster_ids = sorted_centroid_indices[:max_clusters_to_search]
    
        # Step 3: Retrieve candidate vectors from selected clusters (limit candidates per cluster)
        candidates = set()
        max_candidates_per_cluster = 500  # Limit candidates to avoid redundant computations
        for cluster_id in top_cluster_ids:
            cluster_vector_indices = self.cluster_manager.get_vectors_for_cluster(cluster_id)
            candidates.update(cluster_vector_indices[:max_candidates_per_cluster])
    
        candidates = list(candidates)
    
        # Step 4: Preload all candidate vectors in a single read (minimize I/O operations)
        candidate_vectors = np.array([self.get_one_row(idx) for idx in candidates])
        query_norm = np.linalg.norm(query)
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
    
        # Step 5: Compute cosine similarity in batches
        dot_products = np.dot(candidate_vectors, query)
        scores = dot_products / (candidate_norms * query_norm + 1e-10)  # Add epsilon to avoid division by zero
    
        # Step 6: Sort candidates by score and return top-k indices
        sorted_indices = np.argsort(-scores)[:top_k]
        return [candidates[idx] for idx in sorted_indices]


    def get_all_rows(self) -> np.ndarray:
        num_records = os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
        return np.memmap(self.db_path, dtype=np.float32, mode="r", shape=(num_records, DIMENSION))

    def get_one_row(self, row_num: int) -> np.ndarray:
        offset = row_num * DIMENSION * ELEMENT_SIZE
        mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
        return np.array(mmap_vector[0])

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

class ClusterManager:
    def __init__(self, num_clusters: int, dimension: int):
        self.num_clusters = num_clusters
        self.dimension = dimension
        self.centroids = None
        self.assignments = None

    def cluster_vectors(self, vectors: np.ndarray) -> None:
        from faiss import Kmeans  # Import Faiss for k-means

        kmeans = Kmeans(d=self.dimension, k=self.num_clusters, niter=20, seed=DB_SEED_NUMBER)
        batch_size = 500000
        num_vectors = vectors.shape[0]
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            kmeans.train(vectors[start:end].astype(np.float32))

        self.centroids = kmeans.centroids

        # Adjust batch size dynamically for assignment computation
        if num_vectors <= 10**6:
            assignment_batch_size = 10000
        elif num_vectors <= 15 * 10**6:
            assignment_batch_size = 5000
        else:
            assignment_batch_size = 1000

        # Efficient batch assignment
        assignments = np.empty(num_vectors, dtype=np.int32)
        for start in range(0, num_vectors, assignment_batch_size):
            end = min(start + assignment_batch_size, num_vectors)
            batch_distances = np.linalg.norm(vectors[start:end, None] - self.centroids[None, :], axis=2)
            assignments[start:end] = np.argmin(batch_distances, axis=1)

        self.assignments = assignments

    def get_vectors_for_cluster(self, cluster_id: int) -> List[int]:
        return np.where(self.assignments == cluster_id)[0].tolist()
