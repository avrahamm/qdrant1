# Your code here
from qdrant_client import QdrantClient, models
import json
from typing import Any, Generator
import uuid
from qdrant_client.models import PointStruct
from tqdm import tqdm

def stream_json(file_path) -> Generator[Any, Any, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME='arxiv_papers'
# vectors_config={"size": 1536, "distance": "Cosine"}
vectors_config=models.VectorParams(
    size=1536, distance=models.Distance.COSINE
)


if not client.collection_exists(collection_name=f"{COLLECTION_NAME}"):
    client.create_collection(
        collection_name=f"{COLLECTION_NAME}",
        vectors_config=vectors_config,
    )

BATCH_SIZE = 100  # Define your batch size

records_generator = stream_json('data/archive/ml-arxiv-embeddings.json')
points_batch = []

for index, record in tqdm(enumerate(records_generator)):
    embedding = record.get("embedding", None)
    # Skip records with no embedding or invalid embedding
    if embedding is None:
        continue
    if "embedding" in record:
        del record["embedding"]
    point = PointStruct(
        id=str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=record["id"])),
        vector=embedding,
        payload=record,
    )
    points_batch.append(point)

    if len(points_batch) >= BATCH_SIZE:
        client.upsert(
            collection_name=f"{COLLECTION_NAME}",
            points=points_batch,
        )
        print(f"Inserted {len(points_batch)} points, loop index = {index}")
        points_batch.clear()  # Efficiently clear the list

# Process any remaining records in the last batch
if points_batch:
    client.upsert(
        collection_name=f"{COLLECTION_NAME}",
        points=points_batch,
    )

print(f"Inserted last {len(points_batch)} points")
points_batch.clear()  # Efficiently clear the list







