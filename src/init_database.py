"""Initialize database: Vectorize fables and insert into Qdrant"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from src.embeddings import EmbeddingModel
from src.qdrant_manager import QdrantManager
from tqdm import tqdm

# Load environment variables
load_dotenv()


def init_fables_collection():
    """Initialize fables collection"""

    # Configuration from environment variables
    COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'fables')
    DATA_PATH = os.getenv('DATA_PATH', 'data/aesop_fables_processed.json')

    print("=" * 60)
    print("Initializing Fables Vector Database")
    print("=" * 60)

    # 1. Load processed data
    print("\n[1/5] Loading data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        fables = json.load(f)
    print(f"✓ Loaded {len(fables)} fables")

    # 2. Initialize embedding model
    print("\n[2/5] Initializing embedding model...")
    embedding_model = EmbeddingModel()

    # 3. Connect to Qdrant
    print("\n[3/5] Connecting to Qdrant...")
    qdrant = QdrantManager()

    # Delete old collection if exists
    qdrant.delete_collection(COLLECTION_NAME)

    # Create new collection
    vector_dim = embedding_model.get_dimension()
    qdrant.create_collection(COLLECTION_NAME, vector_size=vector_dim)

    # 4. Generate vectors
    print("\n[4/5] Generating vectors...")
    # Use title + content + moral for vectorization
    texts = [
        f"{fable['title']}. {fable['content']} Moral: {fable['moral']}"
        for fable in fables
    ]

    print("  Vectorizing...")
    embeddings = embedding_model.encode(texts, show_progress=True)
    print(f"✓ Generated {len(embeddings)} vectors, dimension: {embeddings.shape[1]}")

    # 5. Insert data into Qdrant
    print("\n[5/5] Inserting data into Qdrant...")

    # Prepare payloads (metadata)
    payloads = [
        {
            'title': fable['title'],
            'content': fable['content'],
            'moral': fable['moral'],
            'language': fable['language'],
            'number': fable['metadata']['number'],
            'word_count': fable['metadata']['word_count']
        }
        for fable in fables
    ]

    # Prepare IDs (extract number from fable_01 as integer ID)
    ids = [int(fable['id'].split('_')[1]) for fable in fables]

    # Insert data
    success = qdrant.insert_vectors(
        collection_name=COLLECTION_NAME,
        vectors=embeddings,
        payloads=payloads,
        ids=ids
    )

    if success:
        print("\n" + "=" * 60)
        print("✓ Initialization complete!")
        print("=" * 60)

        # Show collection info
        info = qdrant.get_collection_info(COLLECTION_NAME)
        if info:
            print(f"\nCollection info:")
            print(f"  Name: {info['name']}")
            print(f"  Vectors count: {info['vectors_count']}")
            print(f"  Points count: {info['points_count']}")
            print(f"  Status: {info['status']}")

        # Test search
        print("\nTesting search functionality...")
        test_query = "a story about honesty and lying"
        print(f"  Query: '{test_query}'")

        query_vector = embedding_model.encode_single(test_query)
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=3
        )

        print(f"\n  Top 3 results:")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result['payload']['title']}")
            print(f"     Similarity score: {result['score']:.4f}")
            print(f"     Moral: {result['payload']['moral']}")

    else:
        print("\n✗ Initialization failed")


if __name__ == '__main__':
    init_fables_collection()
