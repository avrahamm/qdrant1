from qdrant_client import QdrantClient, models
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_embedding(text, model):
    global openai_client
    text = text.replace("\n", "")
    return openai_client.embeddings.create(
        input = [text],
        model=model
    ).data[0].embedding


def find_nearest_points(collection_name, search_vector, limit=5):
    global qdrant_client
    # Query nearest by target_paper_vector
    nearest = qdrant_client.query_points(
        collection_name=collection_name,
        query=search_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return nearest.points


qdrant_client = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI()

if __name__ == '__main__':
    input_text = "the attention mechanism in deep learning"
    model_name = "text-embedding-ada-002"
    embedding = get_embedding(input_text, model_name)

    COLLECTION_NAME = 'arxiv_papers'
    paper_limit = 5
    nearest_points = find_nearest_points(
        collection_name=COLLECTION_NAME,
        search_vector=embedding,
        limit=paper_limit
    )

    nearest_points_ids = []
    for point in nearest_points:
        nearest_points_ids.append(point.payload["id"])
        # print(point.payload["id"])
    print(nearest_points_ids)

