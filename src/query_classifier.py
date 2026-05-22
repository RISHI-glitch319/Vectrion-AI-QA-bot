# =========================================================
# QUERY INTENT CLASSIFIER
# =========================================================

def classify_query(query):

    query = query.lower()

    # =====================================================
    # SUMMARY TYPE
    # =====================================================

    if (
        "summarize" in query
        or "summary" in query
        or "overview" in query
    ):

        return "summary"

    # =====================================================
    # COMPARISON TYPE
    # =====================================================

    elif (
        "compare" in query
        or "difference" in query
        or "vs" in query
    ):

        return "comparison"

    # =====================================================
    # DEFINITION TYPE
    # =====================================================

    elif (
        "define" in query
        or "what is" in query
        or "meaning of" in query
        or "explain" in query
    ):

        return "definition"

    # =====================================================
    # LIST TYPE
    # =====================================================

    elif (
        "list" in query
        or "steps" in query
        or "types" in query
        or "phases" in query
    ):

        return "listing"

    # =====================================================
    # FACTUAL LOOKUP
    # =====================================================

    elif (
        "when" in query
        or "where" in query
        or "who" in query
        or "deadline" in query
    ):

        return "factual"

    # =====================================================
    # GENERAL QUERY
    # =====================================================

    else:

        return "general"