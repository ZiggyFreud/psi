import os
import re
import uuid
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

def clean_response(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'-{2,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def add_to_knowledge(text: str) -> bool:
    """Embed and store a text chunk in ChromaDB. Returns True on success."""
    try:
        doc_id = f"admin_{uuid.uuid4().hex[:12]}"
        collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[{"source": "admin_training"}]
        )
        return True
    except Exception as e:
        print(f"Admin training error: {e}")
        return False

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

Use the provided context as your primary source of information. If the context does not fully cover the question, use your general knowledge about PSI products and wall panel systems to help — but be honest if you are uncertain about specific details. Never fabricate specifications or certifications.

RESPONSE STYLE:
- Write in plain prose only. Absolutely no markdown formatting.
- No bold (**text**), no italic (*text*), no headers (#), no bullet dashes (-), no asterisks (*) of any kind.
- Write everything as natural sentences and paragraphs.
- Do not open with filler phrases like "Great question!" or "I'm on it!"
- PSI does not have architects on staff. Design support means working alongside architects and designers, not providing architectural services directly.
- Keep responses focused and accurate.

CRITICAL — TWO SEPARATE PRODUCT LINES:
PSI makes two completely different product lines. Never mix their specifications.

1. INTERIOR WALL PANELS (Systems 310, 310EB, 312, 314, 410, 412, 631, 632, ClicWall):
   - Designed for commercial and institutional interior use
   - Fire tested to ASTM E84, available in Class A and Class B
   - Use aluminum divider moldings specific to each system
   - Panel thickness 7/16 inch (11.1mm) for most systems; 3/8 inch (10mm) for ClicWall

2. FIPRO MARINE WALL PANELS (completely separate product):
   - Designed exclusively for ships and offshore platforms
   - Fire tested to SOLAS and IMO FTP Code — NOT ASTM E84
   - Core thicknesses: 10mm, 12mm, 16mm, 18-20mm
   - Finished system thickness: 12mm up to 50+mm depending on fire rating and build-up
   - Fire ratings: B-0, B-15, A-15, A-60
   - Panel types: FIPRO and FIPRO MS (lighter weight variant)
   - Certified by Bureau Veritas, MED, USCG, Transport Canada, Lloyd's Register
   - Different moldings and installation requirements than interior panels
   - Never say FIPRO follows ASTM E84 — it does not

CONTACT: Always include BOTH when directing someone to contact PSI:
Email: info@panelspec.com
Phone: 1-800-947-9422

If you truly cannot answer based on the context or your knowledge of PSI, respond with ONLY this exact message:
\"{FALLBACK}\"
Do NOT combine the fallback with any other information.""",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
        ]
    )

    return clean_response(response.content[0].text)
