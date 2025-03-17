from typing import Optional, List, Dict
from pydantic import BaseModel

class SearchRequest(BaseModel):
    """
    Model representing a search request sent by a client.

    Attributes:
        query (str): The search query string provided by the client.
        top_n (Optional[int]): The number of top search results to return.
            Defaults to 5 if not provided.
    """
    query: str
    top_n: Optional[int] = 5

class SearchResult(BaseModel):
    """
    Model representing an individual search result.

    Attributes:
        id (str): Unique identifier for the search result item.
        payload (Dict): A dictionary containing additional data or metadata about the item.
        score (float): The relevance score of the search result, typically used for ranking.
    """
    id: str
    payload: Dict
    score: float

class SearchResponse(BaseModel):
    """
    Model representing the response returned to a client after a search.

    Attributes:
        results (List[SearchResult]): A list containing the top matching search results.
    """
    results: List[SearchResult]