import os
import pathlib
def get_model() -> dict:
    """
    Returns the model name and URL for the OpenAI client.
    """
    model = "gemini-2.5-flash-preview-05-20"  # Local: llama3.2:3b, llama3.2:1b, deepseek-r1:1.5b, dolphin-phi:2.7b; Remote: gemini-2.5-flash-preview-05-20, gemma-3-27b-it
    url = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Local: http://localhost:11434/v1, Remote: https://generativelanguage.googleapis.com/v1beta/openai/
    return {"model":model,"url": url}
def get_api_key() -> str:
    """
    Returns the API key for the gemini client.
    """
    api_key = open(os.path.dirname(__file__)+ "/gemini_api_key.txt", "r").read()
    return api_key