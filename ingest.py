import os
import time
import requests
from bs4 import BeautifulSoup
import voyageai
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

SITEMAPS = [
    "https://panelspec.com/post-sitemap.xml",
    "https://panelspec.com/page-sitemap.xml",
    "https://panelspec.com/portfolio-sitemap.xml",
]

class VoyageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        result = voyage_client.embed(input, model="voyage-2")
        return result.embeddings

chroma_client = chromadb.PersistentClient(path="/data/chroma_db")

try:
    chroma_client.delete_collection("psi_bot")
    print("Deleted old collection.")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="psi_bot",
    embedding_function=VoyageEmbeddingFunction()
)

def get_urls_from_sitemap(sitemap_url):
    urls = []
    try:
        response = requests.get(sitemap_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "xml")
        for loc in soup.find_all("loc"):
            url = loc.text.strip()
            if not url.endswith(('.jpg','.jpeg','.png','.gif','.pdf')):
                urls.append(url)
    except Exception as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
    return urls

def scrape_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Skipped ({response.status_code}): {url}")
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if len(text) > 100:
            return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    return None

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest():
    print("Fetching URLs from sitemaps...")
    all_urls = []
    for sitemap in SITEMAPS:
        urls = get_urls_from_sitemap(sitemap)
        print(f"Found {len(urls)} URLs in {sitemap}")
        all_urls.extend(urls)

    print(f"Total URLs to scrape: {len(all_urls)}")

    all_chunks = []
    all_ids = []
    all_metadata = []

    for url in all_urls:
        print(f"Scraping: {url}")
        text = scrape_page(url)
        if text:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{url}_{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append({"url": url})
        time.sleep(0.5)

    print(f"Total chunks to embed: {len(all_chunks)}")

    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        batch_metadata = all_metadata[i:i + batch_size]
        collection.add(
            documents=batch_chunks,
            ids=batch_ids,
            metadatas=batch_metadata
        )
        print(f"Ingested batch {i // batch_size + 1}")
        time.sleep(1)

    print(f"Ingestion complete! Total chunks: {collection.count()}")

if __name__ == "__main__":
    ingest()
