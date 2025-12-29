"""Qdrant database management module"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QdrantManager:
    """Qdrant vector database manager"""

    def __init__(self):
        """
        Initialize Qdrant client

        Args:
            host: Qdrant server address
            port: Qdrant server port
        """

        QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
        QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"✓ Connected to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Create collection

        Args:
            collection_name: Collection name
            vector_size: Vector dimension
            distance: Distance metric (COSINE, EUCLID, DOT)

        Returns:
            Whether creation was successful
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name in collection_names:
                print(f"⚠ Collection '{collection_name}' already exists")
                return False

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            print(f"✓ Created collection: {collection_name}")
            return True

        except Exception as e:
            print(f"✗ Failed to create collection: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """Delete collection"""
        try:
            self.client.delete_collection(collection_name)
            print(f"✓ Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to delete collection: {e}")
            return False

    def insert_vectors(
        self,
        collection_name: str,
        vectors: List,
        payloads: List[Dict],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Insert vector data

        Args:
            collection_name: Collection name
            vectors: List of vectors
            payloads: List of payload data (metadata)
            ids: List of IDs (optional, auto-generated if not provided)

        Returns:
            Whether insertion was successful
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            # Create points
            points = [
                PointStruct(
                    id=id_,
                    vector=vector.tolist() if hasattr(vector, 'tolist') else vector,
                    payload=payload
                )
                for id_, vector, payload in zip(ids, vectors, payloads)
            ]

            # Batch insert
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            print(f"✓ Inserted {len(points)} records into {collection_name}")
            return True

        except Exception as e:
            print(f"✗ Failed to insert data: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search similar vectors

        Args:
            collection_name: Collection name
            query_vector: Query vector
            limit: Number of results to return
            score_threshold: Score threshold (optional)

        Returns:
            List of search results
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )

            return [
                {
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                }
                for result in results
            ]

        except Exception as e:
            print(f"✗ Search failed: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection information"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            print(f"✗ Failed to get collection info: {e}")
            return None


if __name__ == '__main__':
    # Test Qdrant connection
    manager = QdrantManager()

    # Test collection creation
    collection_name = 'test_collection'
    manager.create_collection(collection_name, vector_size=384)

    # Get info
    info = manager.get_collection_info(collection_name)
    if info:
        print(f"\nCollection info:")
        print(f"  Name: {info['name']}")
        print(f"  Vectors count: {info['vectors_count']}")
        print(f"  Status: {info['status']}")

    # Cleanup
    manager.delete_collection(collection_name)
