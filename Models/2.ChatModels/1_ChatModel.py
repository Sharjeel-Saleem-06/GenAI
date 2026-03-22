from json import load
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

ChatModel = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
result  = ChatModel.invoke("What is the capital of France?")

print(result.content)