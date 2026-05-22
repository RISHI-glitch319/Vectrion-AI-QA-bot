from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


def get_retriever(vectorstore, documents):

    # ============================================
    # SEMANTIC RETRIEVER
    # ============================================

    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # ============================================
    # BM25 RETRIEVER
    # ============================================

    bm25_retriever = BM25Retriever.from_documents(
        documents
    )

    bm25_retriever.k = 4

    # ============================================
    # HYBRID RETRIEVER
    # ============================================

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            semantic_retriever,
            bm25_retriever
        ],
        weights=[0.6, 0.4]
    )

    return hybrid_retriever
