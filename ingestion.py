import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
DATA_DIR = os.path.join("data", "knowledge_base")
DB_DIR = os.path.join("data", "chroma_db")

def ingest_data():
    print("📥 Loading documents...")
    
    # THE FIX: Added loader_kwargs={"encoding": "utf-8"} to handle special characters
    loader = DirectoryLoader(
        DATA_DIR, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"} 
    )
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        return

    print(f"📄 Found {len(documents)} documents. Splitting into chunks...")
    
    # Optimized Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} overlapping chunks.")

    print("🧠 Generating embeddings and saving to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Save to disk
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print("✅ Ingestion complete! Database saved to /data/chroma_db")

if __name__ == "__main__":
    ingest_data()