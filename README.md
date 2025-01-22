# Semantic Search Engine with Vectorized Database and IVF+PQ Indexing  

## Overview  

This script implements a semantic search engine built upon a provided framework for vector storage and access. While the **vector generation** and **memory-mapped storage** were pre-provided, this implementation extends it by incorporating **Inverted File Indexing (IVF)** and **Product Quantization (PQ)** for efficient indexing and querying of high-dimensional vectors.  

## Key Contributions  

### 1. **IVF+PQ Indexing**  
This implementation uses a combination of:  
- **Inverted File Indexing (IVF)**: Divides the vector space into clusters for efficient candidate filtering.  
- **Product Quantization (PQ)**: Compresses vectors into compact codes for fast approximate search within clusters.  

#### Core Indexing Methods:  
- **`_build_index(full_rebuild=False)`**  
  Constructs the IVF+PQ index:
  - Clusters vectors into centroids using K-Means.
  - Trains a PQ codebook for each cluster to quantize the vectors.
  - Saves centroids and compressed cluster data for retrieval.  

- **`_train_pq_codebook(cluster_vectors)`**  
  Trains a PQ codebook using K-Means clustering on vectors within a cluster.  

- **`_quantize(codebook, vector)`**  
  Encodes a vector into a PQ code by matching it to the closest centroid in the codebook.  

### 2. **Efficient Retrieval**  
The retrieval process combines fast filtering using IVF with approximate matching via PQ.  

#### Core Retrieval Methods:  
- **`retrieve(query, top_k)`**  
  - Uses IVF to identify the most relevant clusters based on query similarity to centroids.  
  - Applies PQ decoding and scoring within these clusters to refine and rank the results.  

- **`_pq_search(codes, query, top_k, codebook)`**  
  - Reconstructs vectors from PQ codes and ranks them based on similarity to the query.  

### 3. **Dynamic Updates**  
The implementation allows inserting new vectors and updating the index dynamically:  
- **`insert_records(rows)`**  
  - Appends new vectors to the database and updates the IVF+PQ index accordingly.  


## Provided Framework  

The following features were pre-provided to ensure a consistent starting point:  
1. **Vector Generation**:  
   - **`generate_database(size)`** creates a synthetic dataset of random vectors with fixed dimensions (`70`).  
   - **`_write_vectors_to_file(vectors)`** stores these vectors in a memory-mapped binary file.  

2. **Memory-Mapped Storage**:  
   - Efficient storage and retrieval of vectors using NumPy memory mapping.  
   - **`get_all_rows()`** loads all vectors into memory.  
   - **`get_one_row(row_num)`** retrieves a single vector based on its index.  

---

## Workflow  

1. **Initialization**  
   - Create an instance of `VecDB` with the provided database and index paths.  
   - Load or generate a database.  

2. **Indexing**  
   - Build an IVF+PQ index using the `_build_index` method.  
   - Save centroids and compressed cluster data for fast retrieval.  

3. **Querying**  
   - Use the `retrieve` method to find the top-k most similar vectors to a given query.  

4. **Dynamic Updates**  
   - Add new vectors using `insert_records` and rebuild the index dynamically.  

---

## Example Usage  

```python
import numpy as np
from vec_db import VecDB

# Initialize VecDB
db = VecDB(database_file_path="my_database.dat", index_file_path="my_index", new_db=True, db_size=10000)

# Build the index
db._build_index(full_rebuild=True)

# Query the database
query_vector = np.random.rand(1, 70).astype(np.float32)  # Random query vector
top_results = db.retrieve(query_vector, top_k=5)
print("Top 5 Results:", top_results)

# Insert new vectors
new_vectors = np.random.rand(100, 70).astype(np.float32)  # 100 new vectors
db.insert_records(new_vectors)
```

---

## Highlights 

1. **Indexing**  
   - Efficient clustering with FAISS for IVF.  
   - PQ codebook training and quantization for memory-efficient approximate search.  

2. **Retrieval**  
   - Two-stage filtering and ranking: coarse-grained cluster selection followed by fine-grained PQ decoding.  

3. **Scalability**  
   - Designed to handle large datasets efficiently by leveraging IVF+PQ.  
