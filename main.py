from fastapi import FastAPI
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()

# Connect to the ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")

# Use Ollama to generate embeddings
embedding_function = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

# Get the collection containing the knowledge base
collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=embedding_function,
)


@app.get("/ask")
def ask(question: str):
    # Step 1: RETRIEVE
    # Search ChromaDB for the most relevant documents
    results = collection.query(
        query_texts=[question],
        n_results=2,
    )

    documents = results.get("documents", [[]])[0]

    # Handle case where nothing was retrieved
    if not documents:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the knowledge base.",
            "context_used": [],
        }

    # Combine retrieved documents
    context = "\n\n".join(documents)

    # Step 2: AUGMENT
    augmented_prompt = f"""Use the following context to answer the question.

If the context doesn't contain relevant information, say so.

Context:
{context}

Question:
{question}

Answer clearly and concisely:"""

    # Step 3: GENERATE
    # Send the augmented prompt to the local Ollama model
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {
                "role": "user",
                "content": augmented_prompt,
            }
        ],
    )

    # Return the generated answer and retrieved context
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": documents,
    }