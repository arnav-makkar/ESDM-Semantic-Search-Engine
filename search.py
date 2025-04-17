import os
import re
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --- Step 1: PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Step 2: Text Chunking ---
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits the text into overlapping chunks of 'chunk_size' words with 'overlap'.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# --- Step 3: Embedding Generation with bge-base-en ---
model = SentenceTransformer("BAAI/bge-base-en")

def embed_texts(texts):
    """
    Converts a list of texts to embeddings, ensuring they are float32 for FAISS.
    """
    embeddings = model.encode(texts, convert_to_tensor=False)
    return np.array(embeddings).astype("float32")

# --- Step 4: Create FAISS Index ---
def create_faiss_index(embeddings):
    """
    Creates and returns a FAISS Index using L2 distance.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# --- Helper: Extract Solar Panel Company Name ---
def extract_company_name(text):
    """
    Attempts to extract a solar panel company name from text.
    This heuristic looks for a pattern where a capitalized word (or two)
    is followed by the word 'Solar'.
    """
    pattern = r"([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)?\s+Solar)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return "Unknown"

# --- Step 5: Ingest PDFs and Prepare Data ---
pdf_folder = "solar_pdfs"  # Folder containing your 10 solar panel PDF files.
document_chunks = []       # List for storing text chunks of all PDFs.
metadata = []              # List for storing metadata for each chunk.
file_to_company = {}       # Dictionary mapping filename -> company name

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print("Processing:", filename)
        # Extract all text from the PDF.
        text = extract_text_from_pdf(pdf_path)
        # Extract company name from the full text.
        company = extract_company_name(text)
        file_to_company[filename] = company
        # Chunk the text.
        chunks = chunk_text(text)
        document_chunks.extend(chunks)
        # For each chunk, store metadata (filename in this case).
        metadata.extend([{"source": filename}] * len(chunks))

# --- Step 6: Generate Embeddings and Build FAISS Index ---
embeddings = embed_texts(document_chunks)
index = create_faiss_index(embeddings)
print("FAISS index built with", index.ntotal, "chunks.")

# --- Step 7: Aggregated PDF Ranking Search Function ---
def search_pdfs(query, index, document_chunks, metadata):
    """
    Searches for the query across all chunks, aggregates the results by PDF file,
    and returns a sorted ranking (best match first) based on the lowest distance score.
    """
    # Retrieve all chunks (dataset is small, so we use len(document_chunks) as top_k)
    top_k = len(document_chunks)
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)
    
    # Group results by source file and record the best (lowest) distance for each.
    results_by_pdf = {}
    for dist, idx in zip(distances[0], indices[0]):
        pdf_source = metadata[idx]["source"]
        # Update if this PDF hasn't been seen or if this chunk has a better score.
        if pdf_source not in results_by_pdf or dist < results_by_pdf[pdf_source]["distance"]:
            results_by_pdf[pdf_source] = {"distance": dist, "snippet": document_chunks[idx]}
    
    # Sort the aggregated results by distance (lower is better).
    sorted_results = sorted(results_by_pdf.items(), key=lambda item: item[1]["distance"])
    return sorted_results

# --- Example Query Execution ---
if __name__ == "__main__":
    query = "Efficiency ratings and temperature coefficients for rooftop solar panels"
    aggregated_results = search_pdfs(query, index, document_chunks, metadata)
    
    print("\n--- PDF Ranking Results ---")
    rank = 1
    for pdf_source, data in aggregated_results:
        company = file_to_company.get(pdf_source, "Unknown")
        print(f"Rank {rank}:")
        print("Source File:", pdf_source)
        print("Company:", company)
        print("Best Distance Score:", data["distance"])
        print("Snippet:", data["snippet"][:200])
        print("=" * 50)
        rank += 1
