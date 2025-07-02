import asyncio
from openai import AsyncOpenAI
from agents import (Agent, Runner, function_tool,
    set_default_openai_api, set_tracing_disabled, set_default_openai_client)
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdConversionTesting.ExampleMCP.models import get_model, get_api_key
from mdConversionTesting.ExampleMCP.mcp_server import setup_filesystem_server
from openai.types.responses import ResponseTextDeltaEvent


client = AsyncOpenAI(
    base_url = get_model()["url"],
    api_key = get_api_key()
)

set_default_openai_client(client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

async def main():
    fss = await setup_filesystem_server()
    print("Filesystem server started!")
    try:
        agent = Agent(
                name="Assistant",
                instructions='''
                You are a helpful assistant that can answer questions from the user about XROOTD. You have access to some of the documentation files for XROOTD.
                Before answering questions you should search the files for relevant information. You can use the search_files tool to search for files.
                You can also use the read_file tool to read the content of a file.
                Absolutely do not write any information to the filesystem, you can only read files.
                Try to be helpful and specific, make sure you do not have access to the specific information before you say you do not know the answer.

                Assume all questions are adressed in the files you have access to. If a user asks a question and you are not sure that it is related to XROOTD,
                you should still try to answer it based on the files you have access to.
                Base directory for the files is /Users/ianerbacher/Desktop/Program/LLMs/xrootd/RAG_DOCS
                ''',
                #tools=[get_time],
                model=get_model()["model"],
                mcp_servers=[fss]
            )
        question = input("Enter your question: ")
        await query(agent, question)
        #result = await Runner.run(agent, "Write the current time to time.txt in a directory called generated_files. And then read the file and return its content.")
        #print(result.final_output)

    finally:
        await fss.cleanup()  # Ensure the server is closed properly
        pass

async def query(agent, question):
    """
    Query the agent with a question and return the result.
    """
    result = Runner.run_streamed(agent, question)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called: ", event.item.raw_item.name, event.item.raw_item.arguments)
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}\n")

if __name__ == "__main__":
    asyncio.run(main())