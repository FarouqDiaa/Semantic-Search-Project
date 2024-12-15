# Semantic Search Engine with Vectorized Databases
This repository contains the code and documentation for a simple semantic search engine with vectorized databases and the evaluation of its performance. The project focuses on building an efficient indexing system to retrieve information based on vector space embeddings.

## Project Overview

The key components of the project include:
- `VecDB`: A class representing the vectorized database, responsible for storing and retrieving vectors.
- `generate_database()`: A method to generate a random database.
- `get_one_row()`: A method to get one row from the database given its index.
- `insert_records()`: A method to insert multiple records into the database. It then rebuilds the index.
- `retrieve()`: A method to retrieve the top-k most similar based on a given query vector.
- `_cal_score()`: A helper method to calculate the cosine similarity between two vectors.
- `_build_index()`: A placeholder method for implementing an indexing mechanism.

## Getting Started

To get started with the project, follow these steps:
1. Clone the repository to your local machine.
2. Run the provided code (which is almost the worst implementation of DB) and then edit/update the VecDB class as per the project requirements.
3. Customize the code and add any additional features as needed. However, avoid modifying the DB_SEED_NUMBER, or the VecDB.\_\_init__ & VecDB.retrieve method signature.
4. Run the evaluation to assess the accuracy of your implementation. The final evaluation will use the 'eval' function, but I wil update the 'run_queries' function.

## Usage

The project provides a `VecDB` class that you can use to interact with the vectorized database. Here's an example of how to use it:

```python
import numpy as np
from vec_db import VecDB

# Create an instance of VecDB and random DB of size 10K
db = VecDB(db_size = 10**4)

# Retrieve similar images for a given query
query_vector = np.random.rand(1,70) # Query vector of dimension 70
similar_images = db.retrieve(query_vector, top_k=5)
print(similar_images)
```

Feel free to customize the code to suit your specific use case.
