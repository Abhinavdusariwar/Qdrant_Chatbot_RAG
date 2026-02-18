from pdf import text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from groq import Groq
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from dotenv import load_dotenv
from datasets import Dataset
from langchain_groq import ChatGroq
import os

load_dotenv()

g_client = Groq(api_key = os.getenv("GROQ_API_KEY"))

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 100)
split = splitter.split_text(text)

client = QdrantClient(
    url="https://444d543e-d2ee-4060-ad43-fb65b65285e0.us-east-1-1.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.YXwRMD8gUEyAh7LilFfeWDCh1WaNA-zIU0FlkdnfoGY",
)


client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(split, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True )

from qdrant_client.models import PointStruct
from uuid import uuid4
point = []
for split, vector in zip(split, embeddings):
    point.append(
        PointStruct(
            id = str(uuid4()), 
            vector = vector.tolist(), 
            payload = {
                "text" : split, 
                "source" : "demo"
            }
        )
    )
client.upsert(
    collection_name = "documents", 
    points = point
)

query = "What is MLops?"
query_e = model.encode(query, show_progress_bar=True, normalize_embeddings=True).tolist()

result = client.query_points(collection_name = "documents", query = query_e, limit = 3)
for point in result.points:
    print(point.payload["text"])
    print("Score:", point.score)

co = [point.payload["text"] for point in result.points]
context = "\n\n".join(co)
prompt = f""" 
your a AI Assissant, Answer the question using only the provided context. give me details on these
If the answer is not in the context, say you don't know.
context: {context}
question:{query}"""

response = g_client.chat.completions.create(
    model="llama-3.1-8b-instant", messages=[{'role':'user', 'content':prompt}], temperature = 0.2
)
model_answer =response.choices[0].message.content
print(model_answer)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
    
)



data = {
    "user_input": [query],
    "response": [model_answer],
    "retrieved_contexts": [co],
    "reference": ["MLOps formalizes the lifecycle of machine learning models, borrowing principles from DevOps while addressing ML-specific challenges"]
}
dataset = Dataset.from_dict(data)
res = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy, context_precision, 
        context_recall], llm=llm, embeddings = model, raise_exceptions = False
    )
print(res)

