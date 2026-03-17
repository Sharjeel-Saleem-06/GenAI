import langchain
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ADD THIS LINE TO DEBUG

# Test just the Endpoint first
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

# Use invoke on the LLM directly (instead of the Chat wrapper)
result = llm.invoke("What is the capital of France?")
print(result)
print(f"Token loaded: {hf_token[:5]}..." if hf_token else "Token NOT found!")
print(langchain.__version__)