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

    def get_cluster_assignments(self, cluster_id: int, chunk_size: int = 1000000) -> np.ndarray:
    
        assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")
        if not os.path.exists(assignments_path):
            raise FileNotFoundError("Assignments file not found.")
        
        # Get the total number of assignments from the file size
        total_assignments = os.path.getsize(assignments_path) // np.dtype(np.int32).itemsize
        
        indices = []
        with open(assignments_path, "rb") as f:
            for start in range(0, total_assignments, chunk_size):
                end = min(start + chunk_size, total_assignments)
                f.seek(start * np.dtype(np.int32).itemsize)
                chunk = np.frombuffer(f.read((end - start) * np.dtype(np.int32).itemsize), dtype=np.int32)
                indices.extend(np.where(chunk == cluster_id)[0] + start)  # Offset by start for global indices
        
        return np.array(indices)

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

        query = query.squeeze()
        if query.ndim != 1 or query.shape[0] != DIMENSION:
            raise ValueError(f"Query shape is invalid: {query.shape}. Expected shape is ({DIMENSION},)")

        query_norm = np.linalg.norm(query)
        centroid_distances = np.sum((centroids - query) ** 2, axis=1)
        top_cluster_ids = np.argsort(centroid_distances)[:top_k * 2]
        del centroids
        gc.collect()

        candidate_indices = set()
        for cluster_id in top_cluster_ids:
            candidate_indices.update(self.get_cluster_assignments(cluster_id))
        candidate_indices = np.array(list(candidate_indices))
        db_size = os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
        candidate_indices = candidate_indices[candidate_indices < db_size]

        top_candidates = []
        batch_size = 100000

        for start in range(0, len(candidate_indices), batch_size):
            end = min(start + batch_size, len(candidate_indices))
            batch_indices = candidate_indices[start:end]

            batch_vectors = np.array([self.get_one_row(idx) for idx in batch_indices])
            scores = np.dot(batch_vectors, query) / (
                np.linalg.norm(batch_vectors, axis=1) * query_norm + 1e-10
            )

            for score, idx in zip(scores, batch_indices):
                if len(top_candidates) < top_k:
                    heapq.heappush(top_candidates, (score, idx))
                else:
                    heapq.heappushpop(top_candidates, (score, idx))

        # Step 4: Return sorted results
        top_candidates = sorted(top_candidates, key=lambda x: -x[0])
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
