from sentence_transformers import CrossEncoder


# =========================================================
# LOAD RERANKER MODEL
# =========================================================

reranker_model = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# =========================================================
# RERANK DOCUMENTS
# =========================================================

def rerank_documents(query, documents, top_k=3):

    # Create query-document pairs

    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    # Generate relevance scores

    scores = reranker_model.predict(pairs)

    # Combine documents with scores

    scored_docs = list(zip(documents, scores))

    # Sort by highest relevance

    ranked_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    # Return top documents

    top_documents = [
        doc for doc, score in ranked_docs[:top_k]
    ]

    return top_documents