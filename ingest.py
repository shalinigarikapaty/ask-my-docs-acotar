from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from config import AWS_REGION, EMBEDDING_MODEL, CHROMA_DIR

# Load PDF
loader = PyPDFLoader("data/ACOTAR_MistAndFury.pdf")
documents = loader.load()
print(f"Pages loaded: {len(documents)}")

# Chunk it
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# Connect to Amazon Titan on AWS
embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL,
    region_name=AWS_REGION
)

# Store vectors in ChromaDB locally
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print("Done! Vectors saved to ./chroma_db")