import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY, model="text-embedding-3-small", timeout=120
)
pc = Pinecone(PINECONE_API_KEY)


def get_vector_store():
    try:
        index = pc.Index("codecademy-assesment")
        print("==== Found index ====")
        return index
    except Exception as e:
        print(f"Failed to connect to vector DB index: {e}")
        print("==== Creating new vector DB index ====")
        index = pc.create_index(
            name="codecademy-assesment",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            metric="cosine",
        )
        print("Index created")
        return index


def initialize_vector_db():
    vector_index = get_vector_store()
    if vector_index.describe_index_stats()["total_vector_count"] == 0:
        print("==== Index is empty ====")
        print("==== Loading PDF ====")
        print("==== Embedding and adding documents to index ====")
        loader = PyPDFLoader("data/Nexus Bank Terms and Conditions.pdf")
        docs = loader.load()
        text_split = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_split.split_documents(docs)
        print(f"==== Adding {len(all_splits)} documents to index ====")
        vector_store = PineconeVectorStore(index=vector_index, embedding=embeddings)
        vector_store.add_documents(all_splits)
        print("==== Documents added to index ====")
    else:
        print("==== Index is not empty ====")
        vector_store = PineconeVectorStore(index=vector_index, embedding=embeddings)
    return vector_store
