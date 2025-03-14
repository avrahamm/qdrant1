from qdrant_client import QdrantClient, models
from openai import OpenAI
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def extract_author_name(query_str):
    author_match = re.match(r".*by\s+([A-Za-z\s\-]+)", query_str)
    author = author_match.group(1) if author_match else None
    return author


def get_embedding(text, model):
    global openai_client
    text = text.replace("\n", "")
    return openai_client.embeddings.create(
        input = [text],
        model=model
    ).data[0].embedding


def find_nearest_points(collection_name, search_vector, author=None, limit=5):
    global qdrant_client
    # Create filter if author is specified
    filter_param = None
    if author:
        filter_param = models.Filter(
            must=[
                models.FieldCondition(
                    key='authors',
                    match=models.MatchText(text=author)
                )
            ]
        )

    nearest = qdrant_client.query_points(
        collection_name=collection_name,
        query=search_vector,
        query_filter=filter_param,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return nearest.points


qdrant_client = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI()

if __name__ == '__main__':
    target_text = "Mentions of point clouds by Tian-Xing Xu"
    author_name = extract_author_name(target_text)
    model_name = "text-embedding-ada-002"
    embedding = get_embedding(target_text, model_name)

    COLLECTION_NAME = 'arxiv_papers'
    paper_limit = 3
    nearest_points = find_nearest_points(
        collection_name=COLLECTION_NAME,
        search_vector=embedding,
        author=author_name,
        limit=paper_limit
    )

    nearest_points_ids = []
    for point in nearest_points:
        nearest_points_ids.append(point.payload["id"])
        # print(point.payload["id"])
    print(nearest_points_ids)

