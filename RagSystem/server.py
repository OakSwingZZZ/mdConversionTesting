# Using the Google GenAI API
from google import genai
from google.genai import types
import os
import chromadb
from pathlib import Path
import hashlib
import readline
from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings
import tiktoken
from fastmcp import FastMCP


def get_embedding_models(client):
    """
    Gets a list of all models that currently support embedding content or text.
    """
    for m in client.models.list():
        models = []
        if 'embedContent' in m.supported_actions or 'embedText' in m.supported_actions:
            models.append(m.name)
        return models


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client: genai.Client, model_id: str = "text-embedding-004"):
        """
        Initializes the GeminiEmbeddingFunction with a Google GenAI client and model ID.
        """
        self.client = client
        self.model_id = model_id

    def __call__(self, input: Documents) -> Embeddings:
        #EMBEDDING_MODEL_ID = "models/embedding-001"  # @param ["models/embedding-001", "models/text-embedding-004", "models/gemini-embedding-exp-03-07", "models/gemini-embedding-exp"] {"allow-input": true, "isTemplate": true}
        title = "Custom query"
        response = self.client.models.embed_content(
            model=self.model_id,
            contents=input,
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                title=title
            )
        )
        return response.embeddings[0].values

def create_chroma_db(documents, metadatas, name, client: genai.Client, chroma_dir, model_id="text-embedding-004"):
    from chromadb.config import Settings

    chroma_client = chromadb.PersistentClient(
        path=chroma_dir
    )

    db = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction(client, model_id)
    )

    for doc, meta in zip(documents, metadatas):
        doc_id = hash_doc(doc, meta)
        
        # Check if the ID already exists in the DB
        if not db.get(ids=[doc_id])["ids"]:
            db.add(documents=[doc], ids=[doc_id], metadatas=[meta])  # embedding happens here

    return db

def find_all_files(root_dir):
    return [str(path) for path in Path(root_dir).rglob('*.md') if path.is_file()]

def hash_doc(doc: str, metadata: dict) -> str:
    """
    Generate a unique hash for a document based on its content and metadata.
    """
    combined = doc + str(metadata)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()

def get_relevant_passage(query, db, results=1):
        passage = db.query(query_texts=[query], n_results=results)['documents'][0][0]
        return passage

def split_text_with_tiktoken(text, chunk_size=512, chunk_overlap=100):
    """
    Splits text into chunks using TikToken with a specified chunk size and overlap.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))

    return chunks

def start_mcp_server(chroma_dir, documents_dir, chosen_model, client):
    # Setup the documents
    if not os.path.exists(chroma_dir):
        a = input(f"The chroma directory {chroma_dir} does not exist. Do you want to create it? (y/n): ")
        if a.lower() != 'y':
            print("Exiting the program.")
            exit(1)
        os.makedirs(chroma_dir)

    # Setup documents to embed
    paths = find_all_files(documents_dir)
    if not paths:
        print("No markdown files found in the documents directory.")
        exit(1)
    documents = []
    metadatas = []

    for path in paths:
        with open(path, 'r') as f:
            contents = f.read()
        # LOCATION TO ADD MORE METADATA!!
        if contents.strip():
            # Title could be the filename or extracted from the first Markdown heading
            fileTitle = Path(path).stem  # e.g., "xrootd_config" from "xrootd_config.md"
            # Optionally extract from content: e.g., first line that starts with '# '
            title = None
            for line in contents.splitlines():
                if line.strip().startswith("# "):
                    title = line.strip("# ").strip()
                    break

            # Split the document into chunks using TikToken
            chunks = split_text_with_tiktoken(contents)
            for chunk in chunks:
                documents.append(chunk)
                metadatas.append({"document_title": title, "file_title": fileTitle})

    if documents.__len__() == 0:
        print("No documents found to embed.")
        exit(1)

    # Ensure the chroma_dir exists
    if not os.path.exists(chroma_dir):
        os.makedirs(chroma_dir)

    # Start the vector database
    db = create_chroma_db(
        documents=documents,
        name="xrd_test",
        client=client,
        model_id=chosen_model,
        metadatas=metadatas,
        chroma_dir=chroma_dir
    )

    # Start the MCP server
    server = FastMCP()

    @server.tool
    def get_relevant_passage_tool(query: str, results: int = 1) -> str:
        '''
        Tool to retrieve relevant passage from the database on Xrootd. Change results to get more than one passage. 
        '''
        print(f"Received query: {query}, for tool: get_relevant_passage_tool")
        passage = get_relevant_passage(query, db, results=results)
        return passage if passage else "No relevant passage found."

    return server


# if __name__ == "__main__":
#     # # Set up the Google GenAI client
#     # client = init_client()
#     # # Call the main function with the settings and client
#     # main(chroma_dir, documents_dir, chosen_model, client)




