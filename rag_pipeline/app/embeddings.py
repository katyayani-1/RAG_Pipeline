from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
def generate_embeddings(chunks):
    return model.encode(chunks)

# Initialize FAISS vector index
def initialize_index(dimension):
    return faiss.IndexFlatL2(dimension)

# Add embeddings to the index
def add_to_index(index, embeddings):
    index.add(np.array(embeddings))

# Search the vector index for similar embeddings
def search_index(index, query_embedding, k=5):
    distances, indices = index.search(np.array(query_embedding), k)
    return indices
