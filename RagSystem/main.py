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
from langchain_text_splitters import CharacterTextSplitter


# Settings:
chosen_model = "text-embedding-004"
chroma_dir = os.path.dirname(__file__) + "/chroma_db"
documents_dir = os.path.dirname(__file__) + "/documents"



def init_client():
    """
    Initializes the Google GenAI client.
    """
    try:
        client = genai.Client()
    except Exception as e:
        print("Error initializing Google GenAI client. Please ensure that you have stored the API key in the environment variable GOOGLE_API_KEY.")
        print(e)
        exit(1)
    return client

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
    
def create_chroma_db(documents, name, client: genai.Client, model_id="text-embedding-004"):
    from chromadb.config import Settings

    chroma_client = chromadb.PersistentClient(
        path=chroma_dir
    )

    db = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction(client, model_id)
    )

    for d in documents:
        doc_id = hash_doc(d)
        
        # Check if the ID already exists in the DB
        if not db.get(ids=[doc_id])["ids"]:
            db.add(documents=[d], ids=[doc_id])  # embedding happens here

    return db

def find_all_files(root_dir):
    return [str(path) for path in Path(root_dir).rglob('*.md') if path.is_file()]

def hash_doc(doc: str) -> str:
    return hashlib.sha256(doc.encode('utf-8')).hexdigest()

def get_relevant_passage(query, db):
        passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
        return passage

def query_model(query, passage, client: genai.Client):
    answer = client.models.generate_content(
        model = "gemini-2.5-flash-preview-05-20",
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant that can answer questions from the user about XROOTD. You have access to some of the relevant documentation files for XROOTD. They will be provided to you."),
        contents = [
            query,
            passage
        ]
    )
    print(answer.text)

def logic_loop(db, client: genai.Client):
    """
    Continuously prompts the user for a question and retrieves the relevant passage from the database.
    """
    while True:
        query = input("> Enter your question to retrieve a document,\n> To query Gemini 2.5 Flash with this question and document, add an exclamation point '! [query]'\n> To exit, type 'exit':\n> ")
        if query.lower() == 'exit':
            break
        passage = get_relevant_passage(query, db)
        if passage:
            if query.startswith('!'):
                query_model(query.strip("!"), passage, client)
            else:
                print(f"Relevant passage: \n{passage}")
        else:
            print("No relevant passage found.")

if __name__ == "__main__":
    # Set up the Google GenAI client
    client = init_client()
    # Get the list of embedding models
    embedding_models = get_embedding_models(client)

    # Setup the documents
    if not os.path.exists(chroma_dir):
        a = input(f"The chroma directory {chroma_dir} does not exist. Do you want to create it? (y/n): ")
        if a.lower() != 'y':
            print("Exiting the program.")
            exit(1)
        os.makedirs(chroma_dir)

    #setup documents to embed
    paths = find_all_files(documents_dir)
    if not paths:
        print("No markdown files found in the documents directory.")
        exit(1)
    documents = []
    '''text_splitter = CharacterTextSplitter(separator="\n", 
                                      chunk_size=1000, 
                                      chunk_overlap=100)'''
    for path in paths:
        contents = open(path, 'r').read()
        if contents:
            documents.append(contents)
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
        model_id=chosen_model
    )
    logic_loop(db, client)
    


