import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
import faiss


DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70


class VecDB:
    def __init__(self, database_file_path="saved_db_100.dat", index_file_path="saved_db_100", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.cluster_manager = None
        #self.pq_codebooks = {}  # For Product Quantization
        #self.last_indexed_row = 0 

        # Ensure the index directory exists
        os.makedirs(self.index_path, exist_ok=True)

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        #else:
            #self.load_indices()

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index(full_rebuild=True)  # Full rebuild for a new database

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode="w+", shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def load_indices(self) -> None:
        centroids_path = os.path.join(self.index_path, "ivf_centroids.npy")
        assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")

        if os.path.exists(centroids_path) and os.path.exists(assignments_path):
            # Load centroids and assignments
            centroids = np.load(centroids_path)
            assignments = np.load(assignments_path)

            self.cluster_manager = ClusterManager(num_clusters=len(centroids), dimension=DIMENSION)
            self.cluster_manager.centroids = centroids
            self.cluster_manager.assignments = assignments
        else:
            raise FileNotFoundError("Centroids or assignments files not found.")

        # Load cluster data for each cluster
        self.pq_codebooks = {}
        for cluster_id in np.unique(self.cluster_manager.assignments):
            cluster_file = os.path.join(self.index_path, f"cluster_{cluster_id}.npz")
            if os.path.exists(cluster_file):
                cluster_data = np.load(cluster_file)
                self.pq_codebooks[cluster_id] = {
                    "ids": cluster_data["ids"],
                    "codes": cluster_data["codes"],
                    "codebook": cluster_data["codebook"]
                }
            else:
                print(f"{cluster_file}")
                print(f"Warning: Cluster file for cluster {cluster_id} not found.")



    def _build_index(self, full_rebuild=False):
        vectors = self.get_all_rows()
        

        if full_rebuild:
            num_clusters=max(1, min(len(vectors), int(np.sqrt(len(vectors) / 2)))) // 2
            print(f"num_clusters = {num_clusters}")
            self.cluster_manager = ClusterManager(
                num_clusters, dimension=DIMENSION
                #num_clusters = max(1, min(len(vectors), int(np.sqrt(len(vectors)) * 2))), dimension=DIMENSION
            )
            self.cluster_manager.cluster_vectors(vectors)

            # Save centroids and assignments to disk
            centroids_path = os.path.join(self.index_path, "ivf_centroids.npy")
            assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")
            # TODO: changed
            compressed_centroids = (self.cluster_manager.centroids * 255).astype(np.uint8) 
            np.save(centroids_path, compressed_centroids)
            # np.save(centroids_path, self.cluster_manager.centroids)

            np.save(assignments_path, self.cluster_manager.assignments)

            # Create codebooks and save IDs with PQ codes
            for cluster_id in np.unique(self.cluster_manager.assignments):
                cluster_vector_indices = np.where(self.cluster_manager.assignments == cluster_id)[0]
                cluster_vectors = vectors[cluster_vector_indices]

                # Train PQ codebook
                codebook = self._train_pq_codebook(cluster_vectors)
                pq_codes = np.array([self._quantize(codebook, vec) for vec in cluster_vectors])

                # Save cluster data: IDs, PQ codes, and codebook
                cluster_data = {
                    "ids": cluster_vector_indices,
                    "codes": pq_codes,
                    "codebook": codebook
                }
                cluster_file = os.path.join(self.index_path, f"cluster_{cluster_id}.npz")
                np.savez_compressed(cluster_file, **cluster_data)
        else:
            # Incremental indexing
            new_vectors = vectors[self.last_indexed_row:]
            if len(new_vectors) == 0:
                return  # Nothing to index

            new_assignments = self.cluster_manager.kmeans.predict(new_vectors)

            # Update assignments
            self.cluster_manager.assignments = np.concatenate(
                [self.cluster_manager.assignments, new_assignments]
            )

            # Process each affected cluster
            for cluster_id in np.unique(new_assignments):
                # Combine existing and new vectors for this cluster
                cluster_vector_indices = np.where(self.cluster_manager.assignments == cluster_id)[0]
                cluster_vectors = vectors[cluster_vector_indices]

                # Train PQ codebook
                codebook = self._train_pq_codebook(cluster_vectors)
                pq_codes = np.array([self._quantize(codebook, vec) for vec in cluster_vectors])

                # Save updated cluster data
                cluster_data = {
                    "ids": cluster_vector_indices,
                    "codes": pq_codes,
                    "codebook": codebook
                }
                cluster_file = os.path.join(self.index_path, f"cluster_{cluster_id}.npz")
                np.savez_compressed(cluster_file, **cluster_data)

                # Update PQ codebook in memory
                self.pq_codebooks[cluster_id] = codebook

            self.last_indexed_row = len(vectors)


    def retrieve(self, query: np.ndarray, top_k: int) -> List[int]:
            # Initialize cluster manager and PQ codebooks if not already loaded
        if self.cluster_manager is None:
            self.cluster_manager = None
            self.pq_codebooks = {}
            self.last_indexed_row = 0
            self.load_indices()
        # Step 1: Calculate cosine similarity with cluster centroids
        cluster_scores = [(i, self._cal_score(query, centroid)) for i, centroid in enumerate(self.cluster_manager.centroids)]
        sorted_clusters = sorted(cluster_scores, key=lambda x: -x[1])

        # Step 2: Select top clusters to search within
        # top_cluster_ids = [cluster_id for cluster_id, _ in sorted_clusters[:max(50, top_k * 8)]]
        top_cluster_ids = [cluster_id for cluster_id, _ in sorted_clusters[:max(10, top_k * 8)]]


        # Step 3: Retrieve candidate vectors using PQ scores
        candidates = []
        for cluster_id in top_cluster_ids:
            # Get indices of vectors in this cluster
            cluster_vector_indices = self.cluster_manager.get_vectors_for_cluster(cluster_id)
            cluster_data = np.load(os.path.join(self.index_path, f"cluster_{cluster_id}.npz"))

            # Retrieve PQ codes and codebook
            pq_codes = cluster_data["codes"]
            codebook = cluster_data["codebook"]

            # Perform PQ search to find top candidates
            pq_results = self._pq_search(pq_codes, query, top_k * 15, codebook)
            for idx, pq_score in pq_results:
                candidates.append((cluster_vector_indices[idx], pq_score))  # Map back to original indices

        # Step 4: Retrieve original vectors and re-rank by full similarity
        final_candidates = []
        for idx, _ in candidates:
            original_vector = self.get_one_row(idx)
            score = self._cal_score(query, original_vector)
            final_candidates.append((idx, score))

        # Step 5: Sort candidates by the recalculated similarity score and return top_k
        final_candidates.sort(key=lambda x: -x[1])
        return [int(idx) for idx, _ in final_candidates[:top_k]]

    def retrieve_true(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        num_records = self._get_num_records()
        # here we assume that the row number is the ID of each vector
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]

    def _pq_search(self, codes: np.ndarray, query: np.ndarray, top_k: int, codebook: np.ndarray) -> List[tuple]:
        # Reconstruct vectors using the codebook
        try:
            reconstructed_vectors = np.array([codebook[code] for code in codes])
        except IndexError as e:
            print(f"Error in reconstructing vectors: {e}")
            raise
        query = query.flatten()
        # Calculate scores for each vector against the query using `_cal_score`
        scores = [self._cal_score(reconstructed_vec, query) for reconstructed_vec in reconstructed_vectors]

        # Get the top-k indices with the highest similarity scores
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Return the top-k results as tuples of (index, similarity score)
        return [(idx, scores[idx]) for idx in top_indices]





    def _quantize(self, codebook: np.ndarray, vector: np.ndarray) -> int:
        dot_products = np.dot(codebook, vector)  # Compute dot products for all centroids
        norms = np.linalg.norm(codebook, axis=1) * np.linalg.norm(vector)
        similarities = dot_products / norms
        return int(np.argmax(similarities))



    def _train_pq_codebook(self, cluster_vectors: np.ndarray) -> np.ndarray:
        # num_clusters = max(1, min(len(cluster_vectors) // 5, 4096))
        # TODO: changed
        num_clusters = max(1, min(len(cluster_vectors), int(np.sqrt(len(cluster_vectors) / 4)))) 
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=DB_SEED_NUMBER)
        kmeans.fit(cluster_vectors)
        return kmeans.cluster_centers_

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
        return np.memmap(self.db_path, dtype=np.float32, mode="r", shape=(num_records, DIMENSION))
    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            total_rows = self._get_num_records()
            raise RuntimeError(f"Failed to retrieve row {row_num} out of {total_rows} rows: {e}")
    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity        
class ClusterManager:
    def __init__(self, num_clusters: int, dimension: int):
        self.num_clusters = num_clusters
        self.dimension = dimension
        self.kmeans = None
        self.centroids = None
        self.assignments = None

    def cluster_vectors(self, vectors: np.ndarray) -> None:
        # Ensure data is in float32 format (required for FAISS)
        vectors = vectors.astype(np.float32)

        # Initialize FAISS KMeans
        kmeans = faiss.Kmeans(
            d=self.dimension,      # Vector dimensionality
            k=self.num_clusters,   # Number of clusters
            niter=20,              # Number of iterations
            seed=DB_SEED_NUMBER
        )

        # Train FAISS KMeans
        kmeans.train(vectors)
        self.centroids = kmeans.centroids

        # Use FAISS IndexFlatL2 to assign clusters
        index = faiss.IndexFlatL2(self.dimension)
        index.add(self.centroids)
        _, self.assignments = index.search(vectors, 1)
        self.assignments = self.assignments.flatten()
    def get_vectors_for_cluster(self, cluster_id: int) -> List[int]:
        return np.where(self.assignments == cluster_id)[0]