\# Multi-Document RAG Specification: Gemini 3.0 Edition

:: \*System Goal\*: Build a high-performance RAG system using Gemini 3.0 Flash for speed and deep reasoning.



\## 1. Setup Requirements

The system requires Python 3.10+ and specific Google Generative AI integrations.



\- \*Libraries\*: Install the core dependencies in your terminal.

&nbsp; : `pip install langchain langchain-community langchain-chroma pypdf langchain-google-genai`

\- \*API Key\*: Secure a Gemini 3.0 API key from Google AI Studio.



\## 2. Technical Architecture

The RAG pipeline follows a 6-step logical flow, optimized for the Gemini 3.0 "Thinking" process.







\### Step 1: Document Ingestion

Loads PDF files from the local `./data` directory within your project folder.



\### Step 2: Semantic Chunking

Splits documents into 1000-character segments with a 200-character overlap to preserve sentence context.



\### Step 3: Vector Embeddings

Uses `models/text-embedding-004` to convert text into high-dimensional vectors for semantic search.



\### Step 4: Vector Storage (ChromaDB)

Stores embeddings in a local database to avoid re-processing files on every run.



\## 3. Complete Implementation Code

Save this entire block as `rag\_document/main.py`.



```python

import os

from langchain\_community.document\_loaders import PyPDFLoader

from langchain\_text\_splitters import RecursiveCharacterTextSplitter

from langchain\_google\_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain\_chroma import Chroma

from langchain\_core.prompts import ChatPromptTemplate

from langchain\_core.runnables import RunnablePassthrough

from langchain\_core.output\_parsers import StrOutputParser



\# --- CONFIGURATION ---

os.environ\["GOOGLE\_API\_KEY"] = "YOUR\_GEMINI\_3\_API\_KEY"

FOLDER\_PATH = "data"

CHROMA\_PATH = "chroma\_db"



def load\_documents(folder\_path):

&nbsp;   if not os.path.exists(folder\_path):

&nbsp;       os.makedirs(folder\_path)

&nbsp;       return \[]

&nbsp;   docs = \[]

&nbsp;   for f in os.listdir(folder\_path):

&nbsp;       if f.endswith(".pdf"):

&nbsp;           loader = PyPDFLoader(os.path.join(folder\_path, f))

&nbsp;           docs.extend(loader.load())

&nbsp;   return docs



def split\_text(documents):

&nbsp;   splitter = RecursiveCharacterTextSplitter(chunk\_size=1000, chunk\_overlap=200)

&nbsp;   return splitter.split\_documents(documents)



def create\_vector\_store(chunks):

&nbsp;   embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

&nbsp;   return Chroma.from\_documents(

&nbsp;       documents=chunks,

&nbsp;       embedding=embeddings,

&nbsp;       persist\_directory=CHROMA\_PATH,

&nbsp;       collection\_name="rag\_docs"

&nbsp;   )



def query\_rag\_system(query\_text, vector\_store):

&nbsp;   # Gemini 3.0 Flash with Thinking enabled

&nbsp;   llm = ChatGoogleGenerativeAI(

&nbsp;       model="gemini-3-flash-preview", 

&nbsp;       temperature=1.0, 

&nbsp;       thinking\_level="medium" # Gemini 3.0 specific: minimal, low, medium, or high

&nbsp;   )

&nbsp;   

&nbsp;   retriever = vector\_store.as\_retriever(search\_kwargs={"k": 3})

&nbsp;   prompt = ChatPromptTemplate.from\_template("""

&nbsp;   You are a professional research assistant.

&nbsp;   Answer ONLY using the provided context. 

&nbsp;   If you cannot find the answer, say "I don't know."



&nbsp;   Context: {context}

&nbsp;   Question: {question}

&nbsp;   """)



&nbsp;   chain = (

&nbsp;       {"context": retriever | (lambda docs: "\\n\\n".join(d.page\_content for d in docs)), 

&nbsp;        "question": RunnablePassthrough()}

&nbsp;       | prompt | llm | StrOutputParser()

&nbsp;   )

&nbsp;   return chain.invoke(query\_text)



def main():

&nbsp;   if not os.path.exists(CHROMA\_PATH):

&nbsp;       print("Building Vector DB...")

&nbsp;       docs = load\_documents(FOLDER\_PATH)

&nbsp;       if not docs: 

&nbsp;           print("Please add PDFs to the /data folder.")

&nbsp;           return

&nbsp;       vector\_store = create\_vector\_store(split\_text(docs))

&nbsp;   else:

&nbsp;       embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

&nbsp;       vector\_store = Chroma(persist\_directory=CHROMA\_PATH, embedding\_function=embeddings, collection\_name="rag\_docs")



&nbsp;   while True:

&nbsp;       query = input("\\nAsk (or 'exit'): ")

&nbsp;       if query.lower() == "exit": break

&nbsp;       print("Reasoning with Gemini 3.0...")

&nbsp;       print(f"\\nAnswer: {query\_rag\_system(query, vector\_store)}")



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()





here is the api [REMOVED FOR SECURITY]

