import os
from agents.mcp import MCPServerStdio

async def setup_filesystem_server():
    # Replace './sample_files' with the actual path to your resource directory (xrootd location)
    samples_dir = "./RAG_DOCS" #os.path.dirname(os.path.dirname(__file__)) #os.getcwd()
    server = MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", samples_dir], #Use a prebuilt filesystem MCP server as an example
        }
    )
    await server.connect()
    return server