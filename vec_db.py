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
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index(full_rebuild=True)

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode="w+", shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

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

        # Step 1: Calculate distances to all centroids
        distances = np.linalg.norm(self.cluster_manager.centroids - query, axis=1)
        sorted_centroid_indices = np.argsort(distances)

        # Step 2: Select top clusters to search
        max_clusters_to_search = max(5, min(len(sorted_centroid_indices), top_k * 8))
        top_cluster_ids = sorted_centroid_indices[:max_clusters_to_search]

        # Step 3: Retrieve candidate vectors from selected clusters
        candidates = set()
        for cluster_id in top_cluster_ids:
            cluster_vector_indices = self.cluster_manager.get_vectors_for_cluster(cluster_id)
            candidates.update(cluster_vector_indices)

        # Step 4: Re-rank candidates based on similarity
        final_candidates = []
        for idx in candidates:
            vector = self.get_one_row(idx)
            score = self._cal_score(query, vector)
            final_candidates.append((idx, score))

        final_candidates.sort(key=lambda x: -x[1])  # Sort by descending similarity
        return [idx for idx, _ in final_candidates[:top_k]]

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
        vectors = vectors.astype(np.float32)
        rng = np.random.default_rng(DB_SEED_NUMBER)

        # Initialize centroids randomly
        centroids = vectors[rng.choice(len(vectors), self.num_clusters, replace=False)]

        for _ in range(15):  # Reduced number of iterations
            # Assign vectors to closest centroids
            distances = np.linalg.norm(vectors[:, None] - centroids[None, :], axis=2)
            assignments = np.argmin(distances, axis=1)

            # Recompute centroids
            for i in range(self.num_clusters):
                cluster_points = vectors[assignments == i]
                if len(cluster_points) > 0:
                    centroids[i] = np.mean(cluster_points, axis=0)

        self.centroids = centroids
        self.assignments = assignments

    def get_vectors_for_cluster(self, cluster_id: int) -> List[int]:
        return np.where(self.assignments == cluster_id)[0]
