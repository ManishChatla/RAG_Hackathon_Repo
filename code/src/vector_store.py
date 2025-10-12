import os
import pickle
import chromadb
from embedder import GeminiEmbedder
from config import Config


class ChromaRAG:
    """
    RAG Retriever using Chroma vector database.

    Features:
    - Persistent Chroma client
    - Load embeddings, metadata, and chunks from pickle files
    - Top-k retrieval with distances for confidence scores
    """

    def __init__(self,
                 db_path=Config.CHROMA_DB,
                 collection_name=Config.collection_name,
                 embeddings_file=Config.EMBEDDINGS_FILE,
                 metadata_file=Config.METADATA_FILE,
                 chunks_file=Config.CHUNK_FILE):
        try:
            self.db_path = db_path
            self.collection_name = collection_name
            self.embeddings_file = embeddings_file
            self.metadata_file = metadata_file
            self.chunks_file = chunks_file

            # Ensure DB folder exists
            if not os.path.exists(self.db_path):
                os.makedirs(self.db_path, exist_ok=True)

            # Initialize persistent Chroma client
            self.client = chromadb.PersistentClient(path=self.db_path)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

            # Check if collection already has data
            existing_data = self.collection.get(include=["documents", "metadatas"])
            if len(existing_data.get("documents", [])) == 0:
                print("No existing data found in Chroma — loading embeddings and adding documents...")
                self._load_and_add_documents()
            else:
                print(f"✅ Found {len(existing_data['documents'])} existing documents in Chroma. Skipping insertion.")

        except FileNotFoundError as e:
            print(f"❌ File not found during initialization: {e}")
        except Exception as e:
            print(f"❌ Error initializing ChromaRAG: {e}")

    def _load_and_add_documents(self):
        """Load pickled embeddings, metadata, and chunks and insert them into Chroma."""
        try:
            if not (os.path.exists(self.embeddings_file) and
                    os.path.exists(self.metadata_file) and
                    os.path.exists(self.chunks_file)):
                raise FileNotFoundError(
                    "Missing embeddings, metadata, or chunks pickle files. Please generate them first."
                )

            # Load files
            with open(self.embeddings_file, "rb") as f:
                embeddings = pickle.load(f)
            with open(self.metadata_file, "rb") as f:
                metadata_list = pickle.load(f)
            with open(self.chunks_file, "rb") as f:
                all_chunks = pickle.load(f)

            if not embeddings or not all_chunks:
                raise ValueError("Loaded embeddings or chunks are empty.")

            # Add to Chroma
            self.collection.add(
                documents=all_chunks,
                embeddings=embeddings,
                metadatas=metadata_list,
                ids=[str(i) for i in range(len(all_chunks))]
            )
            print(f"✅ Successfully inserted {len(all_chunks)} documents into Chroma.")

        except (pickle.UnpicklingError, EOFError) as e:
            print(f"❌ Error loading pickle files: {e}")
        except ValueError as e:
            print(f"❌ Validation error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error while adding documents: {e}")

    def retrieve(self, query_text, top_k):
        """
        Retrieve top_k most relevant chunks for the query_text.
        Returns:
            retrieved_chunks: list of unique document chunks
            full_results: dict containing documents, metadatas, and distances
        """
        try:
            if not query_text or not isinstance(query_text, str):
                raise ValueError("Query text must be a non-empty string.")

            embedder = GeminiEmbedder()
            query_emb = embedder.embed_query(query_text)

            if query_emb is None:
                raise ValueError("Failed to generate embedding for the query.")

            # Query Chroma
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            if not results or "documents" not in results or not results["documents"]:
                print("⚠️ No documents retrieved for the query.")
                return [], {}

            retrieved_chunks = list(set(results["documents"][0]))
            return retrieved_chunks, results

        except ValueError as e:
            print(f"❌ Invalid input or embedding issue: {e}")
            return [], {}
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return [], {}
