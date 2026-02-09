#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

print("=== Testing imports ===")

try:
    print("Loading .env...")
    load_dotenv()
    print("✓ .env loaded")
except Exception as e:
    print(f"✗ Error loading .env: {e}")

try:
    print("Importing helper...")
    from src.helper import download_hugging_face_embeddings
    print("✓ helper imported")
except Exception as e:
    print(f"✗ Error importing helper: {e}")
    sys.exit(1)

try:
    print("Downloading embeddings...")
    embeddings = download_hugging_face_embeddings()
    print("✓ Embeddings downloaded")
except Exception as e:
    print(f"✗ Error downloading embeddings: {e}")
    sys.exit(1)

try:
    print("Importing PineconeVectorStore...")
    from langchain_pinecone import PineconeVectorStore
    print("✓ PineconeVectorStore imported")
except Exception as e:
    print(f"✗ Error importing PineconeVectorStore: {e}")
    sys.exit(1)

try:
    print(f"Connecting to Pinecone index 'medical-chatbot'...")
    os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY", "")
    if not os.environ["PINECONE_API_KEY"]:
        print("⚠ WARNING: PINECONE_API_KEY not set")
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )
    print("✓ Connected to Pinecone")
except Exception as e:
    print(f"✗ Error connecting to Pinecone: {e}")
    sys.exit(1)

try:
    print("Importing Gradio...")
    import gradio as gr
    print("✓ Gradio imported")
except Exception as e:
    print(f"✗ Error importing Gradio: {e}")
    sys.exit(1)

print("\n=== All imports successful! ===")
print("You can now run: python gradio_app.py")
