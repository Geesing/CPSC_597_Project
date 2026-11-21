from io import StringIO
from pathlib import Path
import re

from docx import Document
from pypdf import PdfReader
import spacy
import streamlit as st
import torch
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from yasbd import register_spacy_component

import constants as c
from css import CSS
from searcher import Searcher

st.set_page_config(
    page_title='Searcher / Buscadora',
    layout='wide'
)

# Apply Top-Level CSS
st.html('<style>' + CSS + '</style>')

# Initialize models and cache for reuse
@st.cache_resource(show_spinner='Loading... / Cargando...')
def load_models():
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )

    tokenizer = AutoTokenizer.from_pretrained(
        'jinaai/jina-embeddings-v2-base-es',
        trust_remote_code=True,
        local_files_only=True,
        max_length=8192
    )

    model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v2-base-es',
        trust_remote_code=True,
        local_files_only=True
    ).to(device).eval()

    reranker = AutoModelForSequenceClassification.from_pretrained(
        'jinaai/jina-reranker-v2-base-multilingual',
        dtype="auto",
        trust_remote_code=True,
    ).to(device).eval()

    register_spacy_component()
    spacy_model = spacy.blank('xx')
    spacy_model.add_pipe('yasbd', first=True, config={'lang': 'auto'})

    return {
        'device': device,
        'tokenizer': tokenizer,
        'model': model,
        'reranker': reranker,
        'spacy_model': spacy_model
    }
nlp = load_models()

# Index and cache uploaded file's text for searching
@st.cache_resource(show_spinner='Reading... / Leyendo...')
def init_searcher(text, _nlp):
    return Searcher(
        input_text=text,
        nlp=_nlp['spacy_model'],
        tokenizer=_nlp['tokenizer'],
        model=_nlp['model'],
        max_chunk_size=c.MAX_CHUNK_SIZE,
        reranker=_nlp['reranker'],
        window_offset=c.WINDOW_OFFSET
    )

# Load in text from .txt, .docx, and .pdf, escaping sensitive HTML characters
@st.cache_resource
def load_file(file):
    # Table to escape HTML characters
    escaper = str.maketrans('&<>"\'', '\ue001\ue002\ue003\ue004\ue005')

    if file is not None:
        file_type = Path(file.name).suffix.lower()

        if file_type == '.txt':
            doc = StringIO(file.getvalue().decode('utf-8')).read()
        elif file_type == '.docx':
            docx = Document(file)
            doc = '\n'.join(p.text.translate(escaper) for p in docx.paragraphs)
        elif file_type == '.pdf':
            pdf = PdfReader(file)
            doc = '\n\n'.join(p.extract_text(extraction_mode='layout').translate(escaper) for p in pdf.pages)

        return doc
    else:
        return ''

# Clear all input fields and their downstream dependencies
def clear_input():
    init_searcher.clear()
    load_file.clear()
    st.session_state['query'] = ''

    if 'target_start' in st.session_state:
        st.session_state.target_start = None

    if 'target_end' in st.session_state:
        st.session_state.target_end = None

    if 'res_bounds' in st.session_state:
        st.session_state.res_bounds = []

# Main logic starts here

if 'target_start' not in st.session_state:
    st.session_state.target_start = None

if 'target_end' not in st.session_state:
    st.session_state.target_end = None

if 'res_bounds' not in st.session_state:
    st.session_state.res_bounds = []

# Sidebar is locked open
st.set_page_config(initial_sidebar_state='expanded')

with st.sidebar:
    # Layout: 3 sidebar containers
    search_box = st.container(key='search-box')
    search_results = st.container(key='search-results')
    file_upload = st.container(key='file-upload')

    # Business Logic
    with file_upload:
        file = st.file_uploader(
            label='Upload / Subir',
            help='Upload / Subir',
            label_visibility='hidden',
            type=['txt', 'pdf', 'docx'],
            on_change=clear_input
        )
        doc = load_file(file)
        searcher = init_searcher(doc, nlp)

    with search_box:
        query = st.text_input('Search / Buscar', key='query').strip()
        if len(query) > 0:
            results = searcher.get_ranked_similarities(query, rerank_top_k = c.RERANK_K)
            st.session_state.res_bounds = [res.chunk.char_bounds for res in results]
            st.caption(f'{len(results)} sentence(s) / {len(results)} frase(s)')

    with search_results:
        if not st.session_state.res_bounds:
            st.write('') # Filler to keep container slot reserved

        # List clickable text previews
        for i, bounds in enumerate(st.session_state.res_bounds):
            # Skip very short results
            if bounds[1] - bounds[0] < 5:
                continue

            preview_text = doc[bounds[0]:bounds[1]]

            # When button is clicked, note the boundaries of the text it corresponds to
            if st.button(
                preview_text,
                key=f'jump_to_char_{bounds[0]}'
            ):
                st.session_state.target_start = bounds[0]
                st.session_state.target_end = bounds[1]

# Render the document
if st.session_state.target_start is not None and st.session_state.target_end is not None:
    # Dynamically insert positional highlighting based on selected target boundaries
    html = (
        doc[:st.session_state.target_start] +

        '<mark id="target" style="background-color: #fef08a; color: black; padding: 2px 4px; border-radius: 3px;">' +
        doc[st.session_state.target_start:st.session_state.target_end] +
        '</mark>' +

        doc[st.session_state.target_end:]
    )
else:
    # If the user hasn't selected anything, display the whole doc plainly
    html = doc

# Complete escaping HTML chars into their proper aliases 
mapper = str.maketrans ({
    '\ue001': '&amp;',
    '\ue002': '&lt;',
    '\ue003': '&gt;',
    '\ue004': '&quot;',
    '\ue005': '&#39;'
})
html = html.translate(mapper)

# Render the document HTML in an iframe
st.iframe(
    f"""
    <div style="height: 90%; overflow-y: auto; white-space: pre-line; font-family: sans-serif; line-height: 1.5; font-size: 1.1rem; padding: 15px">
        {html}
    </div>
    <script>
        const el = document.getElementById('target');
        if (el) el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
    </script>
    """
    )