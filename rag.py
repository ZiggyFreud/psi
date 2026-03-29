import os
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

FALLBACK = "This product may not be part of our current offerings as it is not listed on our site. However, you can always call us at 1-800-947-9422."

class VoyageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        result = voyage_client.embed(input, model="voyage-2")
        return result.embeddings

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = chroma_client.get_or_create_collection(
            name="psi_bot",
            embedding_function=VoyageEmbeddingFunction()
        )
    return _collection

def query_rag(user_message):
    collection = get_collection()

    results = collection.query(
        query_texts=[user_message],
        n_results=5
    )

    docs = results["documents"][0] if results["documents"] else []

    if not docs:
        return FALLBACK

    context = "\n\n".join(docs)

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=f"""You are a helpful assistant for Panel Specialties Inc (PSI).
Answer questions based ONLY on the provided context from the website.
Be concise and professional.
If the specific product or topic the user is asking about is NOT found in the context, respond with ONLY this exact message and nothing else:
\"{FALLBACK}\"
Do NOT combine this fallback message with any other information.""",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
        ]
    )

    return response.content[0].text
