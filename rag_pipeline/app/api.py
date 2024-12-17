from flask import Flask, request, jsonify
from web_scraper import scrape_website, chunk_text
from embeddings import initialize_index, add_to_index, generate_embeddings
from query_handler import query_to_embedding, generate_response

# Initialize Flask app
app = Flask(__name__)

# Initialize FAISS index
dimension = 384  # Embedding vector dimension for Sentence-BERT
index = initialize_index(dimension)
chunks = []  # Store chunks globally for simplicity

# Endpoint to scrape a website and store embeddings
@app.route('/ingest-website', methods=['POST'])
def ingest_website():
    data = request.json
    url = data.get('url')
    
    # Scrape website and chunk text
    scraped_text = scrape_website(url)
    global chunks
    chunks = chunk_text(scraped_text)
    
    # Generate embeddings and add to FAISS index
    embeddings = generate_embeddings(chunks)
    add_to_index(index, embeddings)
    
    return jsonify({"message": f"Website data from {url} ingested successfully."})

# Endpoint to query the system
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')
    
    # Generate query embedding and search index
    query_embedding = query_to_embedding(user_query)
    indices = search_index(index, query_embedding)
    
    # Get relevant chunks and generate response
    relevant_chunks = " ".join([chunks[i] for i in indices[0]])
    response = generate_response(relevant_chunks, user_query)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000)
