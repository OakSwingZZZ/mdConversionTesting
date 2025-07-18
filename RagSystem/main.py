import os
import threading
import asyncio
from google import genai
from google.genai import types
import readline
from fastmcp import Client

from server import start_mcp_server

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

async def query_model(query, mcp_client: Client, client: genai.Client):
    answer = await client.aio.models.generate_content(
        model = "gemini-2.5-flash-preview-05-20",
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant that can answer questions from the user about XROOTD. You have access to some of the relevant documentation files for XROOTD. They will be provided to you.",
            tools=[mcp_client.session]),
        contents = [
            query
        ],

    )
    print(answer.text)

async def logic_loop(mcp_client: Client, client: genai.Client):
    """
    Continuously prompts the user for a question and retrieves the relevant passage from the database.
    """
    while True:
        query = await asyncio.to_thread(input, "> Enter your question to retrieve a document,\n> To query Gemini 2.5 Flash with this question and document, add an exclamation point '! [query]'\n> To exit, type 'exit':\n> ")
        if query.lower() == 'exit':
            break
        elif query.startswith('!'):
            await query_model(query.strip("!"), mcp_client, client)
        else:
            relevant_passage = await mcp_client.call_tool('get_relevant_passage_tool', {'query': query})
            print(f"Relevant passage: \n{relevant_passage.content}")


#start server
# Settings:
chosen_model = "text-embedding-004"
chroma_dir = os.path.dirname(__file__) + "/chroma_db"
documents_dir = os.path.dirname(__file__) + "/documents"


def start_mcp_server_in_background(chroma_dir, documents_dir, chosen_model, client):
    """
    Starts the MCP server in a background thread to allow other operations to run concurrently.
    """
    def run_server():
        start_mcp_server(chroma_dir, documents_dir, chosen_model, client)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("MCP server is running in the background.")



# async def main():
#     client = init_client()
#     start_mcp_server_in_background(chroma_dir, documents_dir, chosen_model, client)

#     # Retry mechanism for connecting to the MCP server
#     max_retries = 5
#     for attempt in range(max_retries):
#         try:
#             async with Client("http://0.0.0.0:8000/mcp/") as mcp_client:
#                 print("Connected to MCP server.")
#                 await logic_loop(mcp_client)
#                 break
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed: {e}")
#             if attempt < max_retries - 1:
#                 await asyncio.sleep(2)  # Wait before retrying
#             else:
#                 print("Failed to connect to MCP server after multiple attempts.")
#                 raise

async def main():
    client = init_client()
    start_mcp_server_in_background(chroma_dir, documents_dir, chosen_model, client)
    await asyncio.sleep(5)  # wait a moment for the server to start
    async with Client(
        "http://localhost:8000/mcp",
        #transport="streamable-http"
    ) as mcp_client:
    # wait a moment for the server to start
        await asyncio.sleep(1)
        await logic_loop(mcp_client, client=client)
if __name__ == "__main__":
    asyncio.run(main())