from collections import Counter
import re

def expand_query(query: str, docs, n_terms=10, lambda_param=0.6):


    text = " ".join([doc.page_content for doc in docs])
    tokens = re.findall(r"\w+", text.lower())
    query_tokens = re.findall(r"\w+", query.lower())
    original_tokens = set(query_tokens)

    filtered = [t for t in tokens if t not in original_tokens]
    keywords = [term for term, _ in Counter(filtered).most_common(n_terms)]

    expanded_query = f"{query} {' '.join(keywords)}"
    return expanded_query
