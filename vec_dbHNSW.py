from typing import Dict, List, Annotated
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans # New
from sklearn.cluster import KMeans # New
import hnswlib # New
from joblib import Parallel, delayed # New

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.hnsw_indices = {} # New
        self.cluster_manager = None # New
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
            
         # New
        else:
          self.load_indices()
          self._load_cluster_manager()

        # database is generated but not the indices
        if not self.hnsw_indices:
            self._build_index()
            
    def _load_cluster_manager(self):
        if os.path.exists("ivf_centroids.npy") and os.path.exists("ivf_assignments.npy"):
            centroids = np.load("ivf_centroids.npy")
            assignments = np.load("ivf_assignments.npy")
            self.cluster_manager = ClusterManager(num_clusters=len(centroids), dimension=DIMENSION)
            self.cluster_manager.centroids = centroids
            self.cluster_manager.assignments = assignments
        else:
            raise FileNotFoundError("Cluster manager files (centroids or assignments) not found.")

    def load_indices(self):
        # Load IVF cluster assignments
        if os.path.exists("ivf_assignments.npy"):
            self.cluster_assignments = np.load("ivf_assignments.npy")
        else:
            raise FileNotFoundError("Cluster assignment file ivf_assignments.npy not found.")

        if os.path.exists("ivf_centroids.npy") and os.path.exists("ivf_assignments.npy"):
          centroids = np.load("ivf_centroids.npy")
          assignments = np.load("ivf_assignments.npy")
          self.cluster_manager = ClusterManager(num_clusters=len(centroids), dimension=DIMENSION)
          self.cluster_manager.centroids = centroids
          self.cluster_manager.assignments = assignments
        else:
          raise FileNotFoundError("Cluster manager files (centroids or assignments) not found.")

        self.hnsw_indices = {}
        # Load HNSW indices for each cluster
        for cluster_id in np.unique(self.cluster_assignments):
            hnsw_file = f"hnsw_cluster_{cluster_id}.bin"
            if os.path.exists(hnsw_file):
                hnsw_index = HNSWIndex(DIMENSION, max_elements=100000)
                hnsw_index.load(hnsw_file)  # Load serialized index
                self.hnsw_indices[cluster_id] = hnsw_index
            else:
                print(f"Warning: HNSW index file for cluster {cluster_id} not found. Retrieval might be incomplete.")
        self._remove_invalid_clusters()

    def _remove_invalid_clusters(self):
        if self.cluster_manager is None:
            raise ValueError("Cluster manager is not initialized.")
        # Remove clusters that no longer exist in the assignments
        valid_cluster_ids = np.unique(self.cluster_assignments)
        invalid_clusters = [cluster_id for cluster_id in self.hnsw_indices.keys() if cluster_id not in valid_cluster_ids]

        # Remove invalid clusters' HNSW indices
        for cluster_id in invalid_clusters:
            print(f"Removing invalid cluster {cluster_id} and its associated data.")
            del self.hnsw_indices[cluster_id]

        # Update the cluster manager to only contain valid clusters
        self.cluster_manager.centroids = self.cluster_manager.centroids[np.isin(range(len(self.cluster_manager.centroids)), valid_cluster_ids)]
        self.cluster_manager.assignments = np.array([assignment for assignment in self.cluster_assignments if assignment in valid_cluster_ids])


    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        vectors = self._normalize_vectors(vectors)  # Precompute normalized vectors
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        print("rows shape", rows.shape)
        rows = self._normalize_vectors(rows)
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO might change to call insert in the index, if you need
        self._build_index()
        
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        # Ensure query is 2D by flattening extra dimensions
        # print("vector: ", vectors)
        # print("************************************************")
        # print("before normalization: ", vectors.shape)
        if vectors.ndim == 3 and vectors.shape[0] == 1:
            vectors = vectors[0]  # Reduce first dimension
        elif vectors.ndim > 2:
            vectors = vectors.reshape(-1, vectors.shape[-1])  # Flatten to (N, features)
            print("Shape of vectors: ", vectors.shape)
        # print("Input vectors:", vectors)
        if vectors.ndim == 1:  # Handle 1D arrays
            norms = np.linalg.norm(vectors)
            print("************************************************")
            print("Norm for 1D array:", norms)
            if norms == 0:
                return vectors
            return vectors / norms
        elif vectors.ndim == 2:  # Handle 2D arrays
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) # Norm per row
            #print("Norms for 2D array:", norms)
            norms[norms == 0] = 1  # Prevent division by zero
            return vectors / norms
        else:
            raise ValueError("Input array must be 1D or 2D")

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = int(row_num * DIMENSION * ELEMENT_SIZE)
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve row {row_num}: {e}")

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: np.ndarray, top_k=5) -> List[int]:
        # print("query shape in retrieve function", query.shape)
        # print("self._normalize_vectors(query)", self._normalize_vectors(query))
        # print("-" * 100)
        # print("self._normalize_vectors(np.array([query]))", self._normalize_vectors(np.array([query])))
        # print("*" * 100)
        # query = self._normalize_vectors(np.array([query]))[0]
        query = self._normalize_vectors(np.array([query]))[0]
        valid_clusters = list(self.hnsw_indices.keys())
        if not valid_clusters:
            return []
        cluster_distances = np.linalg.norm(self.cluster_manager.centroids[valid_clusters] - query, axis=1)
        cluster_ids = np.argsort(cluster_distances)[:max(3, top_k // 2)]

        #cluster_ids = np.argsort(np.linalg.norm(self.cluster_manager.centroids - query, axis=1))[:max(3, top_k // 2)]
        #cluster_ids = np.argsort(np.linalg.norm(self.cluster_manager.centroids[valid_clusters] - query, axis=1))[:max(3, top_k // 2)]
        #clusted_id = np.unique(cluster_ids)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(f"Cluster IDs {cluster_ids}")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        results = []
        seen_indices = set()  # Use a set to track indices we've already added

        for cluster_id in cluster_ids:
            if cluster_id in self.hnsw_indices:
              cluster_results = self.hnsw_indices[cluster_id].query(query, k=top_k)
              for idx in cluster_results:
                if idx not in seen_indices:
                    similarity = self._cal_score(query, self.get_one_row(idx))
                    results.append((similarity, idx))
                    seen_indices.add(idx)
              #filtered_results = [(self._cal_score(query, self.get_one_row(idx)), idx) for idx in cluster_results if idx not in seen_indices]
              # Add the indices to the seen set
              #seen_indices.update(idx for _, idx in filtered_results)
              # Extend the results with the filtered results
              #results.extend(filtered_results)

            #else:
              #print(f"Skipping cluster {cluster_id}: No HNSW index found.")
        #results.sort(reverse=True)
        results.sort(reverse=True, key=lambda x: x[0])
        return [idx for _, idx in results[:top_k]]

    def retrieve_from_clusters(self, query: np.ndarray, cluster_ids: List[int], top_k=5) -> List[Dict]:
      """
      Retrieve the top-k most similar vectors from specific clusters.

      Parameters:
          query (np.ndarray): The query vector.
          cluster_ids (List[int]): The IDs of the clusters to search.
          top_k (int): Number of top results to return.

      Returns:
          List[Dict]: A list of dictionaries with 'index' and 'similarity' keys.
      """
      # print("query shape in retrieve_from_clusters function", query.shape)
      query = self._normalize_vectors(np.array([query]))[0]
      print (query)
      results = []

      for cluster_id in cluster_ids:
          if cluster_id in self.hnsw_indices:
              # Query the HNSW index of this cluster
              cluster_results = self.hnsw_indices[cluster_id].query(query, k=top_k)

              for idx in cluster_results:
                  vector = self.get_one_row(idx)
                  print(query)
                  similarity = self._cal_score(query, vector)
                  results.append({'index': idx, 'similarity': similarity})
          else:
              print(f"No HNSW index found for cluster {cluster_id}.")

      # Sort results by similarity in descending order
      results.sort(key=lambda x: x['similarity'], reverse=True)

      # Return top-k results
      return results[:top_k]
      #     return [idx for _, idx in results[:top_k]]

    
    def _cal_score(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
      try:
          vec1 = vec1.flatten()  # Converts (1, 70) to (70,)
          if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
              raise TypeError("Inputs to _cal_score must be numpy arrays.")
          if vec1.shape != vec2.shape:
              raise ValueError(f"Shape mismatch: vec1.shape={vec1.shape}, vec2.shape={vec2.shape}")
          vec1 = vec1.astype(np.float32)
          vec2 = vec2.astype(np.float32)
          return np.dot(vec1, vec2)  # Cosine similarity as vectors are normalized
      except Exception as e:
          raise ValueError(e)
    def _build_index(self) -> None:
        vectors = self.get_all_rows()
        vectors = self._normalize_vectors(vectors)
        self.cluster_manager = ClusterManager(
            num_clusters=max(1, min(len(vectors), int(np.sqrt(len(vectors) / 2)))),
            dimension=DIMENSION
        )
        self.cluster_manager.cluster_vectors(vectors)



        ###########
        # Save IVF index (testing purposes)
        np.save("ivf_centroids.npy", self.cluster_manager.kmeans.cluster_centers_)
        np.save("ivf_assignments.npy", self.cluster_manager.assignments)



        #vectors = self._normalize_vectors(vectors) ######## NORMALIZE BEFORE CLUSTERING

        cluster_vectors = {i: [] for i in range(self.cluster_manager.num_clusters)}
        for idx, cluster_id in enumerate(self.cluster_manager.assignments):
            cluster_vectors[cluster_id].append(vectors[idx])

        # Balance clusters
        #cluster_vectors = self._balance_clusters(cluster_vectors)

        # Build HNSW indices
        self.hnsw_indices = {}
        for cluster_id, cluster_vecs in cluster_vectors.items():
            if len(cluster_vecs) > 0:
                hnsw_index = HNSWIndex(DIMENSION, max_elements=100000)
                # hnsw_index.build(np.array(cluster_vecs))
                hnsw_index.build(self._normalize_vectors(np.array(cluster_vecs))) ################ NORMALIZE
                self.hnsw_indices[cluster_id] = hnsw_index

                ########################
                # Save the HNSW index for this cluster
                hnsw_index.index.save_index(f"hnsw_cluster_{cluster_id}.bin")


    def _balance_clusters(self, cluster_vectors, threshold=10):
      """
      Balance clusters by merging small clusters into larger ones and rebuilding affected indices.

      Parameters:
          cluster_vectors (Dict[int, List[np.ndarray]]): A dictionary mapping cluster IDs to vectors.
          threshold (int): Minimum size for a cluster to remain independent.

      Returns:
          Dict[int, List[np.ndarray]]: The updated cluster_vectors.
      """
      small_clusters = []
      large_clusters = {}

      # Separate small and large clusters
      for cluster_id, vecs in cluster_vectors.items():
          if len(vecs) < threshold:
              small_clusters.extend(vecs)
          else:
              large_clusters[cluster_id] = vecs

      # If there are small clusters, reassign them to nearest large clusters
      if small_clusters:
          for vec in small_clusters:
            closest_cluster = max(
                large_clusters.keys(),
                key=lambda cid: np.dot(vec, np.mean(large_clusters[cid], axis=0))  # Use centroid instead of all vectors
                )
            large_clusters[closest_cluster].append(vec)

      # Rebuild HNSW indices for all clusters
      self.hnsw_indices = {}
      for cluster_id, vecs in large_clusters.items():
          if vecs:  # Ensure the cluster is not empty
              hnsw_index = HNSWIndex(DIMENSION, max_elements=100000)
              hnsw_index.build(self._normalize_vectors(np.array(vecs)))
              self.hnsw_indices[cluster_id] = hnsw_index

      # Update cluster assignments and centroids
      all_vectors = []
      new_assignments = []
      new_centroids = []

      for cluster_id, vecs in large_clusters.items():
          all_vectors.extend(vecs)
          new_assignments.extend([cluster_id] * len(vecs))
          new_centroids.append(np.mean(vecs, axis=0))

      self.cluster_manager.assignments = np.array(new_assignments)
      self.cluster_manager.centroids = np.array(new_centroids)

      return large_clusters



class ClusterManager:
    def __init__(self, num_clusters: int, dimension: int):
        self.num_clusters = num_clusters
        self.dimension = dimension
        self.kmeans = None
        self.centroids = None
        self.assignments = None

    def cluster_vectors(self, vectors: np.ndarray) -> None:
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=DB_SEED_NUMBER, batch_size=1024)
        self.assignments = self.kmeans.fit_predict(vectors)
        self.centroids = self.kmeans.cluster_centers_


# class ClusterManager:
#     def __init__(self, num_clusters, dimension):
#         self.num_clusters = num_clusters
#         self.dimension = dimension
#         self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)  # Set n_init explicitly

#     def cluster_vectors(self, vectors):
#         self.assignments = self.kmeans.fit_predict(vectors)
class HNSWIndex:
    def __init__(self, dimension: int, max_elements=1000000, ef_construction=50, m=16):
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=m)
        # self.index.init_index(ef_construction=ef_construction, M=m)

    def build(self, vectors: np.ndarray) -> None:
        unique_vectors = np.unique(vectors, axis=0)
        x = True
        if unique_vectors.shape[0] < vectors.shape[0]:
          assert x == True, "Duplicates Found!"
        self.index.add_items(unique_vectors)


    def query(self, vector: np.ndarray, k: int) -> List[int]:
        labels, _ = self.index.knn_query(vector, k=k)
        return labels[0]


    ## chatgpt
    def save(self, file_path: str) -> None:
        self.index.save_index(file_path)

    def load(self, file_path: str) -> None:
        self.index.load_index(file_path)


