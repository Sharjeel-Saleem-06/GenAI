import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

documents = [
    "Hello, Virat is a professional cricketer",
    "Hania Amir is a famous actress.",
    "Ali is a famous footballer.",
    "Rahat is a famous speaker",
    "Hania loves basketball"
]

query = "Who is Hania Amir?"

# ✅ ONLY WORKING MODEL (v1 API)
model = "models/gemini-embedding-001"

# Embed documents
doc_embeddings = []
for doc in documents:
    res = client.models.embed_content(
        model=model,
        contents=doc
    )
    doc_embeddings.append(res.embeddings[0].values)

# Embed query
query_res = client.models.embed_content(
    model=model,
    contents=query
)
query_embedding = query_res.embeddings[0].values

# Similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index = np.argmax(scores)

print("Query:", query)
print("Best Match:", documents[index])