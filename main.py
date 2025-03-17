from fastapi import FastAPI
from models import SearchRequest, SearchResult, SearchResponse
from qdrant_utils import extract_author_name, get_embedding, find_nearest_points


app = FastAPI()


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    target_text = request.query
    author_name = extract_author_name(target_text)

    model_name = "text-embedding-ada-002"
    embedding = get_embedding(target_text, model_name)
    COLLECTION_NAME = 'arxiv_papers'
    paper_limit = request.top_n  # Change this from request.limit to request.top_n

    try:
        nearest_points = find_nearest_points(
            collection_name=COLLECTION_NAME,
            search_vector=embedding,
            author=author_name,
            limit=paper_limit
        )

        # Initialize as an instance, not a class reference
        search_results = []
        for point in nearest_points:
            search_results.append(
                SearchResult(id=point.id,
                             payload=point.payload,
                             score=point.score)
            )

        return SearchResponse(results=search_results)
    except Exception as e:
        print(f"ERROR in search endpoint: {str(e)}")
