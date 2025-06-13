from Bio import Entrez
import requests
from bs4 import BeautifulSoup
from enum import Enum
from typing import List, Dict

from ddxdriver.utils import strip_all_lines
from ddxdriver.logger import log

ADDED_EXTRA = 10

class Corpus(Enum):
    PUBMED = "PubMed"
    WIKIPEDIA = "Wikipedia"


def api_search(
    query: str,
    top_k: int = 5,
    corpus_name: str = Corpus.PUBMED.value,
) -> List[Dict[str, str]]:
    if corpus_name == Corpus.PUBMED.value:
        return _search_pubmed(query=query, top_k=top_k)
    elif corpus_name == Corpus.WIKIPEDIA.value:
        return _search_wikipedia(query=query, top_k=top_k)


def format_search_result(result):
    """Format a single result into a string."""
    if not all(key in result for key in ["title", "content"]):
        log.warning("Result formatted incorrectly, returning nothing")
    else:
        return f"Title: {result['title']}\nContent:\n{result['content']}"


def _search_pubmed(
    query: str,
    top_k: int = 5,
    min_abstract_length: int = 100,
    email: str = "your.email@example.com",
) -> List[Dict[str, str]]:
    """
    Search PubMed for a query and return the titles and formatted abstracts of the top k results.
    Skip results with abstracts shorter than a specified length.

    Args:
    - query (str): The search query.
    - top_k (int): The number of top results to return.
    - min_abstract_length (int): The minimum length of the abstract to accept.
    - email (str): Email address to use for Entrez.

    Returns:
    - list of dict: A list of dictionaries with 'title' and 'abstract' for each result.
    """
    Entrez.email = email

    # Modify query to include only free full-text articles
    query = f"{query} AND free full text[sb]"
    # query = f"{query} AND ((pmc cc by-sa license[filter]) OR (pmc cc by license[filter]) OR (pmc cc0 license[filter]))"

    # Search PubMed for the query
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=top_k + ADDED_EXTRA)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    id_list = search_results["IdList"]

    results = []

    # Fetch details for each article by PubMed ID
    for pmid in id_list:
        if len(results) >= top_k:
            break

        fetch_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
        fetch_results = Entrez.read(fetch_handle)
        fetch_handle.close()

        article_results = fetch_results["PubmedArticle"]
        if not article_results:
            continue

        article = article_results[0]
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]

        if not article or not title:
            continue
        # Extract and preserve all sections of the abstract
        abstract_sections = (
            article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
        )

        # Construct the full abstract by concatenating sections with their labels
        abstract = ""
        for section in abstract_sections:
            section_label = section.attributes.get("Label", "") if section.attributes else ""
            abstract += f"{section_label}: {section}\n\n" if section_label else f"{section}\n\n"

        # Skip this abstract if it's too short
        if len(abstract) < min_abstract_length:
            continue

        results.append({"title": title.strip(), "content": abstract.strip()})

    return results


def _search_wikipedia(
    query: str, top_k: int = 5, min_summary_length: int = 100
) -> List[Dict[str, str]]:
    """
    Search Wikipedia for a query and return the titles and summaries of the top k results.
    Skip results with summaries shorter than a specified length.

    Handles cases where the query is too long for the Wikipedia API.

    Args:
    - query (str): The search query.
    - top_k (int): The number of top results to return.
    - min_summary_length (int): The minimum length of the summary to accept.

    Returns:
    - list of dict: A list of dictionaries with 'title' and 'summary' for each result.
    """
    url = "https://en.wikipedia.org/w/api.php"
    max_query_length = 300  # Wikipedia's max length

    # Truncate the query if it's too long
    if len(query) > max_query_length:
        query = query[:max_query_length]
        print(f"Query too long. Truncated to: {query}")

    # Perform the search request
    search_response = requests.get(
        url,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": top_k
            + ADDED_EXTRA,  # Fetch more to compensate for potentially short summaries
            "format": "json",
        },
    )

    search_data = search_response.json()
    search_results = search_data["query"]["search"]

    results = []

    # Fetch the summary for each search result
    for result in search_results:
        if len(results) >= top_k:
            break

        title = result["title"]

        # Fetch the page summary
        page_response = requests.get(
            url,
            params={
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "exintro": True,
                "format": "json",
            },
        )

        page_data = page_response.json()
        page_id = next(iter(page_data["query"]["pages"]))
        page_summary = page_data["query"]["pages"][page_id].get("extract", "No summary available")

        # Remove HTML tags from the summary
        page_summary = BeautifulSoup(page_summary, "html.parser").get_text()

        # Skip this summary if it's too short
        if len(page_summary) < min_summary_length:
            continue

        results.append({"title": title.strip(), "content": page_summary.strip()})

    return results
