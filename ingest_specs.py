import os
import json
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import voyageai
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
CHROMA_PATH = "/data/chroma_db"
SPECS_FILE = "./psi_product_specs.json"

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

class VoyageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        result = voyage_client.embed(input, model="voyage-2")
        return result.embeddings

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="psi_bot",
    embedding_function=VoyageEmbeddingFunction()
)

def build_chunks(data):
    chunks = []
    manufacturer = data["manufacturer"]
    shared = data["shared_specifications"]

    shared_text = f"""PSI Panel Systems - Shared Specifications (applies to all standard systems):
Manufacturer: {manufacturer['name']}
Address: {manufacturer['address']}
Phone: {manufacturer['phone_tollfree']} | {manufacturer['phone_local']}
Email: {manufacturer['email']}
Website: {manufacturer['website']}

Panel Thickness (all standard systems): {shared['panel_thickness']}
Molding Material: {shared['molding_material']}
Panel Core: {shared['panel_core']}
Max Horizontal Panel Length: {shared['max_horizontal_panel_length']}
Warranty: {shared['warranty']}

Fire Ratings Available:
{chr(10).join('- ' + r for r in shared['fire_ratings'])}

Laminate Suppliers: {', '.join(shared['laminate_suppliers'])}

Surface Material Options:
{chr(10).join('- ' + s for s in shared['surface_material_options'])}

Installation Methods:
{chr(10).join('- ' + m for m in shared['installation_methods'])}

Installation Requirements:
- Cutting tools: {shared['installation_requirements']['cutting_tools']}
- Face penetrations: {shared['installation_requirements']['face_penetrations']}
- Subwall flatness: {shared['installation_requirements']['subwall_flatness']}
- Acclimation: {shared['installation_requirements']['acclimation']}
- Moisture barrier: {shared['installation_requirements']['moisture_barrier']}
"""
    chunks.append({
        "id": "shared_specs",
        "text": shared_text,
        "metadata": {"source": "psi_product_specs", "type": "shared_specifications"}
    })

    for system in data["systems"]:
        lines = []
        lines.append(f"PSI Panel System: {system['name']}")
        lines.append(f"System ID: {system['id']}")
        lines.append(f"Description: {system['description']}")
        lines.append(f"Panel Thickness: {system['panel_thickness']}")

        if "panel_height" in system:
            lines.append(f"Panel Height: {system['panel_height']}")
        if "panel_width" in system:
            lines.append(f"Panel Width: {system['panel_width']}")
        if "panel_core" in system:
            lines.append(f"Panel Core: {system['panel_core']}")
        if "surface_material" in system:
            lines.append(f"Surface Material: {system['surface_material']}")
        if "connection_system" in system:
            lines.append(f"Connection System: {system['connection_system']}")
        if "edge_finish" in system:
            lines.append(f"Edge Finish: {system['edge_finish']}")
        if "reveal_style" in system:
            lines.append(f"Reveal Style: {system['reveal_style']}")

        reveals = system.get("reveals", {})
        if reveals.get("horizontal"):
            lines.append(f"Horizontal Reveal: {reveals['horizontal']}")
        if reveals.get("vertical"):
            lines.append(f"Vertical Reveal: {reveals['vertical']}")

        if "installation_orientation" in system:
            lines.append(f"Recommended Installation: {system['installation_orientation']}")
        if "max_horizontal_length" in system:
            lines.append(f"Max Horizontal Panel Length: {system['max_horizontal_length']}")

        fire_ratings = system.get("fire_ratings", [])
        if fire_ratings:
            lines.append("Fire Ratings: " + ", ".join(fire_ratings))

        moldings = system.get("divider_moldings", [])
        if moldings:
            lines.append("Divider Moldings:")
            for m in moldings:
                lines.append(f"  - {m}")

        edge_trims = system.get("edge_trim_options", [])
        if edge_trims:
            lines.append("Edge Trim Options:")
            for e in edge_trims:
                lines.append(f"  - {e}")

        molding_detail = system.get("moldings", {})
        if molding_detail:
            if molding_detail.get("edge_trims_half_inch"):
                lines.append("1/2 inch Edge Trims and Corners:")
                for m in molding_detail["edge_trims_half_inch"]:
                    lines.append(f"  - {m}")
            if molding_detail.get("edge_trims_three_eighths_inch"):
                lines.append("3/8 inch Edge Trims and Corners:")
                for m in molding_detail["edge_trims_three_eighths_inch"]:
                    lines.append(f"  - {m}")
            if molding_detail.get("aluminum_finish_options"):
                lines.append("Aluminum Molding Finish Options: " + ", ".join(molding_detail["aluminum_finish_options"]))

        tech = system.get("technical_data", {})
        if tech:
            lines.append("Technical Data:")
            for k, v in tech.items():
                lines.append(f"  - {k.replace('_', ' ').title()}: {v}")

        install_notes = system.get("installation_notes", [])
        if install_notes:
            lines.append("Installation Notes:")
            for note in install_notes:
                lines.append(f"  - {note}")

        chunks.append({
            "id": f"system_{system['id']}",
            "text": "\n".join(lines),
            "metadata": {"source": "psi_product_specs", "type": "system_spec", "system_id": system["id"]}
        })

    return chunks


def main():
    if not os.path.exists(SPECS_FILE):
        print(f"Specs file not found: {SPECS_FILE}")
        return

    with open(SPECS_FILE, "r") as f:
        data = json.load(f)

    chunks = build_chunks(data)
    print(f"Built {len(chunks)} chunks from specs JSON.")

    try:
        existing = collection.get(where={"source": "psi_product_specs"})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"Removed {len(existing['ids'])} existing spec chunks.")
    except Exception as e:
        print(f"No existing chunks to remove ({e})")

    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

    print(f"Added {len(chunks)} chunks to ChromaDB.")
    print(f"Collection now has {collection.count()} total chunks.")

main()
