import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from google import genai
from google.genai import types

# --- CONFIGURATION ---
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

FOLDER_PATH = "data"
CHROMA_PATH = "chroma_db"

def load_documents(folder_path):
    """Loads all PDFs from the specified folder."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
    docs = []
    for f in os.listdir(folder_path):
        if f.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(folder_path, f))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {f}: {e}")
    return docs

def split_text(documents):
    """Splits documents into manageable chunks for the vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    """Creates or updates the Chroma vector database."""
    # Using text-embedding-004 (Standard for 2025-2026)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="rag_docs"
    )

def query_rag_system(query_text, vector_store):
    """Retrieves context and queries Gemini 3 Flash."""
    # 1. Retrieve relevant docs using LangChain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query_text)
    context = "\n\n".join(d.page_content for d in docs)

    # 2. Initialize the Gemini 3 Client
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    
    # System instructions are the preferred way to set 'Thinking' behavior in Gemini 3
    sys_instruction = (
        "You are a professional research assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say 'I don't know.' "
        "Be concise and factual."
    )

    user_prompt = f"Context: {context}\n\nQuestion: {query_text}"

    try:
        # Use 'gemini-3-flash-preview' for the best balance of speed and reasoning
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=0.3, # Lower temperature for higher factual RAG accuracy
            )
        )
        return response.text
    except Exception as e:
        return f"API Error: {e}"


def refresh_vector_store(vector_store):
    """Adds new documents from the data folder to the vector store."""
    print("Scanning for new documents...")
    docs = load_documents(FOLDER_PATH)
    if not docs:
        print("No PDFs found in data folder.")
        return
    
    # In a production system, we would check for duplicates here.
    # For this simple version, we will just add them. 
    # Chroma handles deduplication of *exact* same IDs, but here we are generating new chunks.
    # A full rebuild is cleaner for this scale.
    
    print(f"Processing {len(docs)} document pages...")
    chunks = split_text(docs)
    vector_store.add_documents(chunks)
    print(f"Added {len(chunks)} new chunks to the database.")

def main():
    # Setup - Always return a usable vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if os.path.exists(CHROMA_PATH):
        print("--- Loading Existing Vector DB ---")
        vector_store = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings, 
            collection_name="rag_docs"
        )
    else:
        print("--- Building Vector DB Subject to Creation ---")
        docs = load_documents(FOLDER_PATH)
        if docs:
            chunks = split_text(docs)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
                collection_name="rag_docs"
            )
            print(f"Index created with {len(chunks)} chunks.")
        else:
            # Create empty if no docs found, so we can add later
            vector_store = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=embeddings, 
                collection_name="rag_docs"
            )
            print("Vector DB initialized (empty). Add files and type 'update'.")

    # Chat loop
    print("\nGemini 3 RAG System Ready.")
    print("Commands: 'exit' to quit, 'update' to re-scan PDFs.")
    
    while True:
        query = input("\nYour Question: ")
        
        if query.lower() in ["exit", "quit"]: 
            break
        elif query.lower() == "update":
            # Smart update: Clear and Rebuild
            print("Rebuilding database to ensure all files are current...")
            
            # 1. Delete existing data on disk to be safe
            if os.path.exists(CHROMA_PATH):
                vector_store.delete_collection()
            
            # 2. Re-instantiate the vector store class completely
            # This fixes "ValueError: Chroma collection not initialized"
            vector_store = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=embeddings, 
                collection_name="rag_docs"
            )
            
            refresh_vector_store(vector_store)
            continue

        print("Reasoning with Gemini 3 Flash...")
        answer = query_rag_system(query, vector_store)
        print(f"\n[Gemini 3 Flash]:\n{answer}")

if __name__ == "__main__":
    main()
