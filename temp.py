import os
import re
import pdfplumber
import numpy as np
import faiss
import streamlit as st
import openai

from sentence_transformers import SentenceTransformer, CrossEncoder
from rapidfuzz import fuzz, process

#######################
# Helper Functions
#######################

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks (by word count)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def extract_company_name(text):
    """
    Extract a solar panel company name from text using a regex pattern and fuzzy match 
    it against a known list.
    """
    # Pattern: Look for one or two capitalized words preceding "Solar"
    pattern = r"([A-Z][a-zA-Z0-9&\s]{0,30}\s+Solar)"
    match = re.search(pattern, text)
    candidate = match.group(1).strip() if match else ""
    known_companies = [
        "Adani Solar",
        "Tata Power Solar",
        "Waaree Energies",
        "Vikram Solar",
        "ReNew Power",
        "Jakson Solar",
        "Loom Solar",
        "Azure Power",
        "Amplus Solar",
        "Havells Solar"
    ]

    if candidate:
        best_match, score, _ = process.extractOne(candidate, known_companies, scorer=fuzz.ratio)
        if score > 50:
            return best_match
        else:
            return candidate
    else:
        return "Unknown"

def expand_query(query):
    """
    Expand the query using GPT (if OPENAI_API_KEY is set), otherwise fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", 
                           "content": f"Expand the following search query with synonyms and related technical concepts to improve recall: '{query}'"}],
                temperature=0.7,
                max_tokens=50
            )
            expanded = response.choices[0].message['content'].strip()
            return expanded
        except Exception as e:
            st.warning(f"OpenAI API error: {e}")
            return query + " efficiency temperature coefficient NOCT"
    else:
        # Fallback expansion if no API key available.
        return query + " efficiency temperature coefficient NOCT"

def highlight_text(text, query_terms):
    """Highlight query terms within the text by wrapping them in <mark> HTML tags."""
    highlighted = text
    for term in query_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", highlighted)
    return highlighted

def get_query_terms(expanded_query):
    """
    Extract significant words from the expanded query.
    This simple version splits on spaces and removes short words.
    """
    stop_words = {"and", "the", "for", "with", "a", "an", "of", "to", "in"}
    terms = [word.strip(".,") for word in expanded_query.split() if len(word) > 2 and word.lower() not in stop_words]
    return list(set(terms))


#######################
# Load and Process PDFs (Cached)
#######################

@st.cache_data(show_spinner=False)
def load_pdf_data(pdf_folder="solar_pdfs"):
    """
    Load PDFs from the folder, extract text, chunk them,
    and collect metadata and company name per file.
    Returns: document_chunks, metadata (list of dicts), file_to_company dict.
    """
    document_chunks = []
    metadata = []
    file_to_company = {}
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            st.write(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            company = extract_company_name(text)
            file_to_company[filename] = company
            chunks = chunk_text(text)
            document_chunks.extend(chunks)
            metadata.extend([{"source": filename}] * len(chunks))
    return document_chunks, metadata, file_to_company

#######################
# Build FAISS Index (Cached)
#######################

@st.cache_resource(show_spinner=False)
def build_index(document_chunks, _model):
    embeddings = _model.encode(document_chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

#######################
# Initialize Models
#######################

# Primary encoder for chunk embeddings using SPECTER2.
@st.cache_resource(show_spinner=False)
def load_primary_model():
    return SentenceTransformer("allenai/specter")

# Cross-encoder for reranking.
@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

primary_model = load_primary_model()
cross_encoder = load_cross_encoder()

#######################
# Search and Rerank Function
#######################

def search_pdfs(query, index, document_chunks, metadata, file_to_company, top_k_initial=50):
    # Expand the query
    # expanded_query = expand_query(query)

    expanded_query = query
    query_terms = get_query_terms(expanded_query)

    # Encode the expanded query using the primary model.
    query_embedding = primary_model.encode([expanded_query], convert_to_tensor=False)
    
    # Perform initial FAISS search over all chunks.
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k_initial)
    
    # Gather candidate snippets.
    candidate_results = []
    for dist, idx in zip(distances[0], indices[0]):
        pdf_source = metadata[idx]["source"]
        snippet = document_chunks[idx]
        candidate_results.append({
            "source": pdf_source,
            "snippet": snippet,
            "faiss_distance": dist
        })
    
    # Prepare pairs for cross-encoder reranking.
    pairs = [(expanded_query, cand["snippet"]) for cand in candidate_results]
    rerank_scores = cross_encoder.predict(pairs)
    # The cross-encoder returns a score (higher means more relevant)
    for i, cand in enumerate(candidate_results):
        cand["rerank_score"] = rerank_scores[i]
    
    # Aggregate best snippet per PDF (use highest rerank_score)
    results_by_pdf = {}
    for cand in candidate_results:
        source = cand["source"]
        score = cand["rerank_score"]
        if source not in results_by_pdf or score > results_by_pdf[source]["rerank_score"]:
            results_by_pdf[source] = cand
    
    # Sort aggregated results by rerank_score (desc order)
    sorted_results = sorted(results_by_pdf.items(), key=lambda x: x[1]["rerank_score"], reverse=True)
    return sorted_results, query_terms, expanded_query

#######################
# Streamlit UI
#######################

st.title("Solar Panel PDF Semantic Search")
st.write("Enter a technical query (e.g., 'efficiency ratings and temperature coefficients for rooftop panels'):")

user_query = st.text_input("Search Query:")
if st.button("Search") and user_query:
    with st.spinner("Processing your query..."):
        # Load PDF data
        document_chunks, metadata, file_to_company = load_pdf_data()
        # Build the FAISS index if not already built.
        index = build_index(document_chunks, _model=primary_model)
        # Search and rerank
        results, query_terms, expanded_query = search_pdfs(user_query, index, document_chunks, metadata, file_to_company)
    
    st.write(f"**Original Query:** {user_query}")
    st.write(f"**Expanded Query:** {expanded_query}")
    st.write("**Matched Query Terms:**", ", ".join(query_terms))
    
    st.markdown("---")
    st.subheader("PDF Ranking Results:")
    if not results:
        st.write("No results found.")
    else:
        for rank, (pdf_source, data) in enumerate(results, start=1):
            company = file_to_company.get(pdf_source, "Unknown")
            # Highlight query terms in the snippet.
            snippet_highlighted = highlight_text(data["snippet"], query_terms)
            st.markdown(f"**Rank {rank}:**")
            st.markdown(f"**Source File:** {pdf_source}")
            st.markdown(f"**Company:** {company}")
            st.markdown(f"**Reranker Score:** {data['rerank_score']:.4f}")
            st.markdown(f"**Snippet:** {snippet_highlighted}", unsafe_allow_html=True)
            st.markdown("---")
