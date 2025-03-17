from qdrant_client import QdrantClient, models
from openai import OpenAI
import re
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

        try:
            nearest = qdrant_client.query_points(
                collection_name=collection_name,
                query=search_vector,
                query_filter=filter_param,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return nearest.points
        except Exception as e:
            print(f"ERROR in find_nearest_points: {str(e)}")


qdrant_client = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI()
