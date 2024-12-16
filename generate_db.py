from vec_db import VecDB
from evaluation import *
import numpy as np
import time

# Function to run queries and store the results
def run_queries(db, all_db, top_k, num_queries):
    results = []
    
    for _ in range(num_queries):
        # Select a random vector from the database as the query
        query_idx = np.random.randint(len(all_db))
        query_vector = all_db[query_idx]
        
        # Perform the retrieval
        start_time = time.time()
        retrieved_indices = db.retrieve(query_vector, top_k=top_k)
        query_time = time.time() - start_time
        
        # Get the actual top-k IDs from the database (simplified ground truth)
        actual_ids = list(range(query_idx - top_k + 1, query_idx + 1))
        
        # Convert retrieved_indices to integers to avoid np.uint64
        retrieved_indices = [int(idx) for idx in retrieved_indices]

        results.append({
            'query_vector': query_vector,
            'retrieved_indices': retrieved_indices,
            'actual_ids': actual_ids,
            'query_time': query_time
        })
    
    return results

# Function to evaluate the results (e.g., average score and runtime)
def eval(results, db):
    total_score = 0
    total_runtime = 0
    for result in results:
        # Compute similarity score between query vector and retrieved vectors
        query_vector = result['query_vector']
        retrieved_indices = result['retrieved_indices']
        
        # Calculate the average similarity score for retrieved vectors
        avg_score = np.mean([np.dot(query_vector, db.get_one_row(idx)) for idx in retrieved_indices])
        total_score += avg_score
        total_runtime += result['query_time']
    
    avg_score = total_score / len(results)
    avg_runtime = total_runtime / len(results)
    return avg_score, avg_runtime

# Main logic
db = VecDB(database_file_path="saved_db.dat", index_file_path="index.dat", db_size=100)  # Use an existing database

# Retrieve all vectors from the database
all_db = db.get_all_rows()

# Run queries and evaluate results
top_k = 5
num_queries = 10

# Run the queries and evaluate the results
results = run_queries(db, all_db, top_k, num_queries)
avg_score, avg_runtime = eval(results, db)

# Print the results
print(f"Average Score: {avg_score}")
print(f"Average Query Runtime: {avg_runtime} seconds")

# Check if scores make sense
print("Query Results:")
for result in results:
    print(f"Top-k IDs (DB): {result['retrieved_indices']}")
    print(f"Top-k IDs (Actual): {result['actual_ids']}")
    
    # Ensure no duplicates in the retrieved indices
    assert len(result['retrieved_indices']) == len(set(result['retrieved_indices'])), \
        f"Duplicate IDs found in result.retrieved_indices: {result['retrieved_indices']}"
    print("-" * 50)
