from embeddings import generate_embeddings, search_index
import openai

# Use your OpenAI API key
openai.api_key = "your-api-key"
# Generate a query embedding
def query_to_embedding(query):
    return generate_embeddings([query])[0]  # Generate embedding for a single query

# Generate a response using OpenAI LLM
def generate_response(context, query):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant for answering user queries based on provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return response['choices'][0]['message']['content']
