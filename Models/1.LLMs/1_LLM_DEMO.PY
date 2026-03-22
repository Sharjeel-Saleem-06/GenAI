import langchain_google_genai
from dotenv import load_dotenv

load_dotenv()

llm = langchain_google_genai.llms.GoogleGenerativeAI(model="gemini-2.5-flash")
print(llm.invoke("What is the capital of France?"))