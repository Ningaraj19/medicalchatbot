import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from pinecone import Pinecone
from helper import extract_from_pdf, filter_to_min, text_split, download_embeddings


load_dotenv()

PINECONE_API_KEY = os.getenv("PINEKONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data=extract_from_pdf("data/")
minimal_docs=filter_to_min(extracted_data)
chunked_texts= text_split(minimal_docs)
embedding = download_embeddings()

pinecone_api = PINECONE_API_KEY
pclient = Pinecone(api_key=pinecone_api)

index_name = "medical-chatbot"
if not pclient.has_index(index_name):
    pclient.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    ) 
index = pclient.Index(index_name)

doc_search = PineconeVectorStore.from_documents(
    documents = chunked_texts,
    embedding=embedding,
    index_name=index_name
)
