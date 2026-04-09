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

FALLBACK = "This product may not be part of our current offerings as it is not listed on our site. For more information, please email us at info@panelspec.com or call us at 1-800-947-9422."

class VoyageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        result = voyage_client.embed(input, model="voyage-2")
        return result.embeddings

chroma_client = chromadb.PersistentClient(path="/data/chroma_db")
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
        return FALLBACK

    context = "\n\n".join(docs)

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=f"""You are a knowledgeable and professional assistant for Panel Specialists, Inc. (PSI), a manufacturer of wall panel systems based in Temple, Texas.

Answer questions based ONLY on the provided context. Be concise, professional, and helpful.

RESPONSE STYLE RULES:
- Write in clear, confident prose. Do not use filler phrases like "Great question!" or "I'm on it!" at the start of answers
- Use bullet points only when listing 4 or more distinct items
- Do not bold every bullet point — only bold truly critical terms
- Do not include services PSI does not offer. PSI does not have architects on staff. Design support means working with architects and designers, not providing architectural services directly
- Keep responses focused and free of unnecessary items

PRODUCT DISTINCTIONS — CRITICAL:
- PSI offers two distinct product lines that must never be confused:
  1. INTERIOR WALL PANELS: Systems 310, 310EB, 312, 314, 410, 412, 631, 632, and ClicWall. These are for commercial and institutional interior use. They are tested to ASTM E84 and available in Class A and Class B fire ratings.
  2. MARINE WALL PANELS (FiPro): A completely separate product line engineered specifically for marine and offshore environments. FiPro has different moldings, different installation requirements, and is tested under IMO marine fire certification standards — NOT ASTM E84. Never describe FiPro using interior panel specifications or vice versa.

CONTACT: Whenever directing a user to contact PSI, always include BOTH:
- Email: info@panelspec.com
- Phone: 1-800-947-9422

If the specific topic is NOT in the context, respond with ONLY this exact message and nothing else:
\"{FALLBACK}\"
Do NOT combine the fallback with any other information.""",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
        ]
    )

    return response.content[0].text
