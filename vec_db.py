import os
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
import heapq
import gc


DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db_100.dat", index_file_path="saved_db_100", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.cluster_manager = None


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

    def load_centroids(self) -> np.ndarray:
        centroids_path = os.path.join(self.index_path, "ivf_centroids.npy")
        if os.path.exists(centroids_path):
            return np.load(centroids_path, mmap_mode='r')
        else:
            raise FileNotFoundError("Centroids file not found.")

    def get_cluster_assignments(self, cluster_ids: List[int]) -> np.ndarray:
        assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")
        if os.path.exists(assignments_path):
            full_assignments = np.load(assignments_path, mmap_mode='r')
            mask = np.isin(full_assignments, cluster_ids)
            indices = np.where(mask)[0]
            return indices
        else:
            raise FileNotFoundError("Assignments file not found.")
        
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
        os.makedirs(self.index_path, exist_ok=True)
        centroids = self.load_centroids()

        # Validate Query
        query = query.squeeze()
        if query.ndim != 1 or query.shape[0] != DIMENSION:
            raise ValueError(f"Query shape is invalid: {query.shape}. Expected shape is ({DIMENSION},)")

        # Step 1: Calculate distances to centroids
        centroid_distances = np.linalg.norm(centroids - query, axis=1)
        centroid_distances = np.argsort(centroid_distances)
        del centroids
        gc.collect()

        # Step 2: Select top clusters
        max_clusters_to_search = min(len(centroid_distances), top_k * 4)
        top_cluster_ids = centroid_distances[:max_clusters_to_search]
        del centroid_distances
        gc.collect()

        # Step 3: Fetch candidate indices
        candidate_indices = self.get_cluster_assignments(top_cluster_ids)
        candidate_indices = np.unique(candidate_indices)
        db_size = os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
        candidate_indices = candidate_indices[candidate_indices < db_size]

        # Step 4: Process candidates in chunks
        query_norm = np.linalg.norm(query)
        top_candidates = []
        chunk_size = 200

        for start in range(0, len(candidate_indices), chunk_size):
            end = min(start + chunk_size, len(candidate_indices))
            chunk_indices = candidate_indices[start:end]

            candidate_vectors = [self.get_one_row(idx) for idx in chunk_indices]
            candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
            dot_products = np.dot(candidate_vectors, query)
            del candidate_vectors

            scores = dot_products / (candidate_norms * query_norm + 1e-10)
            del dot_products, candidate_norms

            for idx, score in zip(chunk_indices, scores):
                if len(top_candidates) < top_k:
                    heapq.heappush(top_candidates, (score, idx))
                else:
                    heapq.heappushpop(top_candidates, (score, idx))
            del chunk_indices
            gc.collect()
        # Step 5: Return sorted results
        top_candidates = sorted(top_candidates, key=lambda x: -x[0])
        gc.collect()
        return [idx for _, idx in top_candidates]


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
        self.assignments_path = None

    def cluster_vectors(self, vectors: np.ndarray) -> None:
        from faiss import Kmeans  # Import Faiss for k-means

        kmeans = Kmeans(d=self.dimension, k=self.num_clusters, niter=20, seed=DB_SEED_NUMBER)
        batch_size = 500000
        num_vectors = vectors.shape[0]
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            kmeans.train(vectors[start:end].astype(np.float32))
        
        self.centroids = kmeans.centroids
        del kmeans
        # batch size dynamically for assignment computation
        # if num_vectors <= 10**6:
        #     assignment_batch_size = 10000
        # elif num_vectors <= 15 * 10**6:
        #     assignment_batch_size = 5000
        # else:
        #     assignment_batch_size = 1000

        assignment_batch_size = max(1000, num_vectors // 10000)

        # Efficient batch assignment
        assignments = np.empty(num_vectors, dtype=np.int32)
        for start in range(0, num_vectors, assignment_batch_size):
            end = min(start + assignment_batch_size, num_vectors)
            batch_distances = np.linalg.norm(vectors[start:end, None] - self.centroids[None, :], axis=2)
            assignments[start:end] = np.argmin(batch_distances, axis=1)
            del batch_distances

        self.assignments = assignments
        del assignments

    def get_vectors_for_cluster(self, cluster_id: int) -> List[int]:
        return np.where(self.assignments == cluster_id)[0].tolist()
