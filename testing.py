import numpy as np
from vec_db import VecDB

# Define the comparison function with debugging
def compare_retrieval_methods(db: VecDB, query: np.ndarray, top_k: int = 5):
    """
    Compare the optimized 'retrieve' method with the brute-force 'retrieve_true' method.
    Display debugging information with cosine similarity scores.

    Parameters:
    - db: Instance of VecDB
    - query: Query vector for retrieval
    - top_k: Number of top results to retrieve and compare
    """
    # Retrieve results from both methods
    ids_retrieve = db.retrieve(query, top_k)
    ids_linear = db.retrieve_true(query, top_k)

    # Debugging: Compute cosine similarity scores for the retrieved results
    print("\n=== Debugging Cosine Similarity Scores ===")

    # Optimized retrieve method
    print("\n[Optimized retrieve method]")
    scores_retrieve = []
    for idx in ids_retrieve:
        vector = db.get_one_row(idx)  # Fetch the vector from DB
        score = db._cal_score(query, vector)  # Compute cosine similarity
        score = float(score)  # Ensure scalar conversion
        scores_retrieve.append((idx, score))
        print(f"ID: {idx}, Cosine Similarity: {score:.6f}")

    # Brute-force retrieve_true method
    print("\n[Brute-force retrieve_true method]")
    scores_linear = []
    for idx in ids_linear:
        vector = db.get_one_row(idx)
        score = db._cal_score(query, vector)
        score = float(score)
        scores_linear.append((idx, score))
        print(f"ID: {idx}, Cosine Similarity: {score:.6f}")

    # Compare the results
    print("\n=== Comparison of Results ===")
    print("Top IDs (retrieve):", ids_retrieve)
    print("Top IDs (retrieve_true):", ids_linear)

    # Find and display common IDs
    common_ids = set(ids_retrieve) & set(ids_linear)
    print(f"Common IDs: {common_ids}")
    print(f"Number of common IDs: {len(common_ids)} out of {top_k}")

    # Calculate precision
    precision = len(common_ids) / top_k
    print(f"Precision: {precision:.2f}")

    # Sort scores for a clearer comparison
    print("\n=== Sorted Cosine Similarity Scores (for comparison) ===")
    scores_retrieve.sort(key=lambda x: -x[1])  # Sort by score descending
    scores_linear.sort(key=lambda x: -x[1])   # Sort by score descending

    # Display sorted scores for Optimized retrieve method
    print("\n[Sorted Optimized retrieve method]")
    for idx, score in scores_retrieve:
        print(f"ID: {idx}, Cosine Similarity: {score:.6f}")

    # Display sorted scores for Brute-force retrieve_true method
    print("\n[Sorted Brute-force retrieve_true method]")
    for idx, score in scores_linear:
        print(f"ID: {idx}, Cosine Similarity: {score:.6f}")


# Main testing script
if __name__ == "__main__":
    # Define the dimension for vectors (ensure it matches VecDB DIMENSION)
    DIMENSION = 70

    # Initialize your VecDB instance
    db = VecDB(database_file_path="saved_db_1m.dat", index_file_path="saved_db_1m", db_size=10**6)

    # Generate a random query vector
    query_vector = np.random.rand(1, DIMENSION).astype(np.float32)

    # Compare the two retrieval methods
    print("=== Comparison of Retrieval Methods ===")
    compare_retrieval_methods(db, query_vector, top_k=5)
