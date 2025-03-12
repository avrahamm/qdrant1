from qdrant_client import QdrantClient, models


def find_nearest_points(collection_name, paper_id, limit=5):
    global client
    # scroll_filter={"must": [{"key": "id", "match": {"value": paper_id}}]}
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(key="id", match=models.MatchValue(value=paper_id)),
        ]
    )
    result = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    records = result[0]

    # print(type(result[0][0].vector))
    # print(result[0][0].vector)

    if not records:
        raise Exception("No results found")

    target_paper_point = records[0]
    # print(target_paper_point)
    # print(type(target_paper_point))
    target_paper_vector = target_paper_point.vector

    # Query nearest by target_paper_vector
    nearest = client.query_points(
        collection_name=collection_name,
        query=target_paper_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return nearest.points


if __name__ == '__main__':
    client = QdrantClient(host="localhost", port=6333)
    COLLECTION_NAME = 'arxiv_papers'
    paper_id = "1311.5068"
    paper_limit = 5
    nearest_points = find_nearest_points(collection_name=COLLECTION_NAME, paper_id=paper_id, limit=paper_limit)

    nearest_points_ids = []

    for point in nearest_points:
        nearest_points_ids.append(point.payload["id"])
        # print(point.payload["id"])

    print(nearest_points_ids)
