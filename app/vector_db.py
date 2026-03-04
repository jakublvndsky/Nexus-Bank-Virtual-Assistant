import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_store = InMemoryVectorStore(embeddings)


def initialize_vector_db():
    loader = PyPDFLoader("data/Nexus Bank Terms and Conditions.pdf")
    docs = loader.load()
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_split.split_documents(docs)
    vector_store.add_documents(all_splits)
    return "Documents stored in vector index"
