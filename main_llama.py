import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
FOLDER_PATH = "data"
CHROMA_PATH = "chroma_db_llama"

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
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    """Creates or updates the Chroma vector database."""
    # Using all-MiniLM-L6-v2 as per the guide
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH,
        collection_name="rag_docs_llama"
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def query_rag_system(query_text, vector_store):
    """Retrieves context and queries Llama 3."""
    llm = ChatOllama(model="llama3") 
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer ONLY using the context below. 
        If the answer is not present, say "I don't know."
        
        Context:
        {context}
        
        Question:
        {question}
        """
    )
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(query_text)

def main():
    # Setup - Always return a usable vector store
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(CHROMA_PATH):
        print("--- Loading Existing Vector DB (Llama) ---")
        vector_store = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding_function, 
            collection_name="rag_docs_llama"
        )
    else:
        print("--- Building Vector DB (Llama) Subject to Creation ---")
        docs = load_documents(FOLDER_PATH)
        if docs:
            chunks = split_text(docs)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=CHROMA_PATH,
                collection_name="rag_docs_llama"
            )
            print(f"Index created with {len(chunks)} chunks.")
        else:
            # Create empty if no docs found
            vector_store = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=embedding_function, 
                collection_name="rag_docs_llama"
            )
            print("Vector DB initialized (empty). Add files and type 'update'.")

    # Chat loop
    print("\nLlama 3 RAG System Ready.")
    print("Commands: 'exit' to quit, 'update' to re-scan PDFs.")
    
    while True:
        query = input("\nYour Question: ")
        
        if query.lower() in ["exit", "quit"]: 
            break
        elif query.lower() == "update":
            print("Rebuilding database...")
            if os.path.exists(CHROMA_PATH):
                vector_store.delete_collection()
            
            # Rebuild
            docs = load_documents(FOLDER_PATH)
            chunks = split_text(docs)
            if chunks:
                vector_store = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embedding_function,
                    persist_directory=CHROMA_PATH,
                    collection_name="rag_docs_llama"
                )
                print(f"Rebuilt with {len(chunks)} chunks.")
            else:
                print("No documents found.")
            continue

        print("Thinking with Llama 3...")
        try:
            answer = query_rag_system(query, vector_store)
            print(f"\n[Llama 3]:\n{answer}")
        except Exception as e:
            print(f"Error communicating with Ollama (is it running? 'ollama run llama3'): {e}")

if __name__ == "__main__":
    main()
