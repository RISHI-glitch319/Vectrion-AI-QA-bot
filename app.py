import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import tempfile
import time

from streamlit_pdf_viewer import pdf_viewer

from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm_pipeline import generate_response
from src.query_classifier import classify_query

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Vectrion AI",
    page_icon="🚀",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "uploaded_pdf_paths" not in st.session_state:
    st.session_state.uploaded_pdf_paths = []

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "memory_context" not in st.session_state:
    st.session_state.memory_context = ""

# =========================================================
# FUTURISTIC CSS
# =========================================================

st.markdown("""
<style>

/* ======================================================
GLOBAL
====================================================== */

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
}

/* ======================================================
BACKGROUND
====================================================== */

.stApp {
    background:
        radial-gradient(circle at top left, rgba(0,255,255,0.15), transparent 30%),
        radial-gradient(circle at top right, rgba(124,58,237,0.18), transparent 30%),
        linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%);
    color: white;
}

/* ======================================================
SIDEBAR
====================================================== */

[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.82);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ======================================================
GLASS PANELS
====================================================== */

.glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px;
    backdrop-filter: blur(20px);
    margin-bottom: 12px;
}

/* ======================================================
PDF VIEWER
====================================================== */

/* ======================================================
CHAT AREA
====================================================== */

.chat-wrapper {
    height: 68vh;
    overflow-y: auto;
    padding-right: 8px;
    border-radius: 16px;
    margin-bottom: 10px;
}

/* ======================================================
USER MESSAGE
====================================================== */

.user-msg {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    padding: 14px;
    border-radius: 16px;
    color: white;
    margin-bottom: 12px;
}

/* ======================================================
ASSISTANT MESSAGE
====================================================== */

.assistant-msg {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px;
    border-radius: 16px;
    margin-bottom: 12px;
}

/* ======================================================
CHAT INPUT
====================================================== */

.stChatInputContainer {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ======================================================
SCROLLBARS
====================================================== */

.chat-wrapper::-webkit-scrollbar,
.pdf-wrapper::-webkit-scrollbar {
    width: 7px;
}

.chat-wrapper::-webkit-scrollbar-thumb,
.pdf-wrapper::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🚀 Vectrion AI")

    st.markdown("---")

    st.markdown("""
    <div class="glass">
    💬 Conversational AI<br><br>
    📄 Multi-PDF Intelligence<br><br>
    🧠 Hybrid Retrieval<br><br>
    🚀 Citation Grounding<br><br>
    ⚡ Enterprise RAG
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🕒 Memory Timeline")

    if len(st.session_state.messages) == 0:

        st.info("No conversations yet.")

    else:

        for msg in st.session_state.messages:

            if msg["role"] == "user":

                st.markdown(
                    f"""
                    <div class="glass">
                    💬 {msg["content"][:60]}...
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================================================
# HEADER
# =========================================================

st.title("🚀 Vectrion AI")

st.caption(
    "Enterprise AI Workspace with Conversational RAG Intelligence"
)

st.markdown("---")

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_files = st.file_uploader(
    "📂 Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

# =========================================================
# PROCESS DOCUMENTS
# =========================================================

if uploaded_files and not st.session_state.documents_processed:

    all_documents = []

    with st.spinner("⚡ Processing Documents..."):

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(uploaded_file.read())

                temp_path = tmp.name

            st.session_state.uploaded_pdf_paths.append(
                temp_path
            )

            docs = load_pdf(temp_path)

            all_documents.extend(docs)

        chunks = split_documents(all_documents)

        embeddings = get_embedding_model()

        vectorstore = create_vector_store(
            chunks,
            embeddings
        )

        retriever = get_retriever(
            vectorstore,
            chunks
        )

        st.session_state.retriever = retriever

        st.session_state.documents_processed = True

    st.success("✅ AI Workspace Ready!")

# =========================================================
# MAIN LAYOUT
# =========================================================

left_col, right_col = st.columns([1, 1])

# =========================================================
# LEFT PDF WORKSPACE
# =========================================================

with left_col:

    st.markdown(
        "<h3>📑 Document Workspace</h3>",
        unsafe_allow_html=True
    )

    if len(st.session_state.uploaded_pdf_paths) > 0:

        pdf_names = [
            os.path.basename(path)
            for path in st.session_state.uploaded_pdf_paths
        ]

        selected_name = st.selectbox(
            "Select Document",
            pdf_names
        )

        selected_index = pdf_names.index(selected_name)

        selected_pdf_path = (
            st.session_state.uploaded_pdf_paths[selected_index]
        )

        with open(selected_pdf_path, "rb") as f:

            pdf_bytes = f.read()

        st.markdown(
            '<div class="pdf-wrapper">',
            unsafe_allow_html=True
        )

        pdf_viewer(
            input=pdf_bytes,
            width="100%",
            height=900,
            render_text=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

# =========================================================
# RIGHT AI CHAT WORKSPACE
# =========================================================

with right_col:

    st.markdown(
        "<h3>🤖 AI Copilot</h3>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="glass">
    ⚡ Hybrid Retrieval Active &nbsp;&nbsp;&nbsp;
    🧠 Conversational Memory Enabled
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # SCROLLABLE CHAT AREA
    # =====================================================

    st.markdown(
        '<div class="chat-wrapper">',
        unsafe_allow_html=True
    )

    for msg in st.session_state.messages:

        if msg["role"] == "user":

            st.markdown(
                f"""
                <div class="user-msg">
                {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="assistant-msg">
                {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # CHAT INPUT
    # =====================================================

    query = st.chat_input(
        "Ask anything about your documents..."
    )

    # =====================================================
    # QUERY HANDLING
    # =====================================================

    if query and st.session_state.retriever:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # =================================================
        # QUERY INTENT CLASSIFICATION
        # =================================================

        query_type = classify_query(query)

        # =================================================
        # THINKING ANIMATION
        # =================================================

        thinking_box = st.empty()

        thinking_steps = [
            "🔍 Hybrid Retrieval Running...",
            "🧠 Searching Semantic Memory...",
            "📄 Grounding Sources...",
            "🚀 Generating Enterprise Response..."
        ]

        for step in thinking_steps:

            thinking_box.markdown(
                f"""
                <div class="glass">
                {step}
                </div>
                """,
                unsafe_allow_html=True
            )

            time.sleep(0.6)

        # =================================================
        # RETRIEVAL
        # =================================================

        retrieved_docs = (
            st.session_state.retriever.invoke(query)
        )

        # =================================================
        # CONTEXT CREATION WITH CITATIONS
        # =================================================

        context = ""

        for i, doc in enumerate(retrieved_docs):

            page_num = doc.metadata.get(
                "page",
                "Unknown"
            )

            context += f"""

Source {i+1} | Page {page_num}

{doc.page_content}

==================================================
"""

        # =================================================
        # GENERATE RESPONSE
        # =================================================

        answer = generate_response(
            query,
            context,
            st.session_state.memory_context
        )

        thinking_box.empty()

        # =================================================
        # STREAMING RESPONSE
        # =================================================

        response_placeholder = st.empty()

        streamed_text = ""

        for char in answer:

            streamed_text += char

            response_placeholder.markdown(
                f"""
                <div class="assistant-msg">
                {streamed_text}▋
                </div>
                """,
                unsafe_allow_html=True
            )

            time.sleep(0.002)

        response_placeholder.markdown(
            f"""
            <div class="assistant-msg">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # SAVE MEMORY
        # =================================================

        st.session_state.memory_context += f"""

User:
{query}

Assistant:
{answer}
"""

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🚀 Vectrion AI • Advanced Enterprise Conversational RAG Workspace"
)