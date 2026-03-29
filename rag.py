import os
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

class VoyageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        result = voyage_client.embed(input, model="voyage-2")
        return result.embeddings

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="psi_bot",
    embedding_function=VoyageEmbeddingFunction()
)

def query_rag(user_message):
    results = collection.query(
        query_texts=[user_message],
        n_results=5
    )

    docs = results["documents"][0] if results["documents"] else []

    if not docs:
        return "We currently may not be offering this product as it is not listed on our site, however you can always call us at: 1-800-947-9422"

    context = "\n\n".join(docs)

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="""You are a helpful assistant for Panel Specialties Inc (PSI).
Answer questions based only on the provided context from the website.
If the answer is not in the context, respond with:
'We currently may not be offering this product as it is not listed on our site, however you can always call us at: 1-800-947-9422'
Be concise and professional.""",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
        ]
    )

    return response.content[0].text
