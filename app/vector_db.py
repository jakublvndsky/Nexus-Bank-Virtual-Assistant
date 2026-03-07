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
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
pc = Pinecone(PINECONE_API_KEY)


def get_vector_store():
    try:
        index = pc.Index("codecademy-assesment")
        print("==== Found index ====")
        return PineconeVectorStore(index=index, embedding=embeddings)
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
        return PineconeVectorStore(index=index, embedding=embeddings)


def initialize_vector_db():
    loader = PyPDFLoader("data/Nexus Bank Terms and Conditions.pdf")
    docs = loader.load()
    vector_store = get_vector_store()
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_split.split_documents(docs)
    vector_store.add_documents(all_splits)
    return vector_store
