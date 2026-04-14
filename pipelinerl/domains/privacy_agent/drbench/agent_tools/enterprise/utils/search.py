"""Search term extraction and relevance scoring utilities"""

import re
from typing import List, Set


def extract_search_terms(query: str) -> List[str]:
    """
    Extract relevant search terms from a query

    Args:
        query: The search query

    Returns:
        List of extracted search terms
    """
    # Common stop words to filter out
    stop_words = {
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "use",
        "our",
        "we",
        "can",
        "how",
        "what",
        "data",
        "information",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "about",
        "all",
        "any",
        "as",
        "if",
        "so",
        "that",
        "this",
        "there",
        "including",
        "related",
        "such",
        # Search-specific terms
        "search",
        "find",
        "look",
        "query",
        "results",
        # Resource-specific terms
        "internal",
        "external",
        "repository",
        "repositories",
        "drbench",
        "document",
        "documents",
        "file",
        "files",
        "documentation",
    }

    # Extract words using regex
    words = re.findall(r"\b\w+\b", query.lower())

    # Filter out stop words and short words
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)

    return unique_keywords


def calculate_relevance_score(content: str, search_terms: List[str]) -> float:
    """
    Calculate relevance score for content based on search terms

    Args:
        content: Content to score
        search_terms: Terms to search for

    Returns:
        Relevance score between 0 and 1
    """
    if not content or not search_terms:
        return 0.0

    content_lower = content.lower()
    total_score = 0.0
    max_possible_score = len(search_terms)

    for term in search_terms:
        term_lower = term.lower()

        # Exact word match (highest score)
        if re.search(r"\b" + re.escape(term_lower) + r"\b", content_lower):
            total_score += 1.0
        # Partial match
        elif term_lower in content_lower:
            total_score += 0.5
        # Fuzzy match (e.g., plural forms)
        elif term_lower[:-1] in content_lower and len(term_lower) > 3:
            total_score += 0.3

    return min(total_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0


def filter_by_relevance(
    items: List[dict], search_terms: List[str], content_key: str = "name", threshold: float = 0.1
) -> List[dict]:
    """
    Filter items by relevance score

    Args:
        items: List of items to filter
        search_terms: Search terms
        content_key: Key in item dict containing searchable content
        threshold: Minimum relevance score

    Returns:
        Filtered and sorted list of items
    """
    scored_items = []

    for item in items:
        content = str(item.get(content_key, ""))
        score = calculate_relevance_score(content, search_terms)

        if score >= threshold:
            item_copy = item.copy()
            item_copy["relevance_score"] = score
            scored_items.append(item_copy)

    # Sort by relevance score (descending)
    scored_items.sort(key=lambda x: x["relevance_score"], reverse=True)

    return scored_items
