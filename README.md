# 🕵️ The Agency: Multi-Model RAG System

**The Agency** is a powerful, flexible Retrieval-Augmented Generation (RAG) system capable of chatting with your documents using state-of-the-art LLMs. This project builds upon the foundational RAG tutorial by [Aman Kharwal](https://thecleverprogrammer.com/2026/01/06/building-a-multi-document-rag-system/) and extends it with production-ready features and dual-model architecture.

## 🎯 My Enhancements

While following Aman Kharwal's excellent RAG tutorial, I've significantly expanded the functionality to create a more robust, production-ready system:

### 🔧 Key Improvements

**1. Upgraded to Gemini 3 Flash Preview**
- Migrated from Llama 3 to Google's cutting-edge Gemini 3 Flash model
- Implemented proper system instructions for enhanced reasoning
- Optimized temperature settings (0.3) for higher factual accuracy
- Integrated the latest `google.genai` SDK with `types.GenerateContentConfig`

**2. Production-Grade Vector Store Management**
- **Persistent Storage:** Vector database automatically saves to disk (`chroma_db/`) - no need to rebuild on every run
- **Smart Update Command:** Built-in `update` command that completely rebuilds the index when new PDFs are added
- **Graceful Error Handling:** Comprehensive try-catch blocks and folder creation logic
- **Empty Store Initialization:** System can start even with no documents and add them later

**3. Enhanced Embedding Strategy**
- Upgraded to `models/text-embedding-004` (Google's latest embedding model)
- Increased retrieval from k=3 to k=5 for better context coverage
- Maintained optimal chunk_size=1000 with overlap=200 for context preservation

**4. Dual-Architecture Design**
- **`main.py`**: Cloud-powered with Gemini 3 Flash (fast, powerful reasoning)
- **`main_llama.py`**: Privacy-focused local processing with Llama 3 via Ollama
- Users can choose their workflow based on privacy needs and performance requirements

**5. Developer Experience**
- Comprehensive error messages with clear guidance
- Auto-folder creation if `data/` doesn't exist
- Environment variable validation on startup
- Clean separation of concerns across functions

## 🚀 Two Workflows, One Repo

| Feature | **Gemini Mode** (`main.py`) | **Llama Mode** (`main_llama.py`) |
|:--------|:----------------------------|:----------------------------------|
| **Model** | **Gemini 3 Flash Preview** | **Llama 3** (local via Ollama) |
| **Speed** | ⚡ Extremely Fast | 🐢 Hardware Dependent |
| **Privacy** | Data sent to Google API | 🔒 100% Local / Private |
| **Setup** | Requires API Key | Requires Ollama installation |
| **Best For** | Complex reasoning, heavy workloads | Privacy-sensitive docs, offline use |
| **Embeddings** | `text-embedding-004` | `all-MiniLM-L6-v2` |
| **Context Retrieval** | Top 5 chunks | Top 3 chunks |

---

## 📺 See It In Action
Check out the demo recording:
🎥 **[Watch the Demo Video](assets/demo.mp4)**

---

## 🛠️ Technical Stack

- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) with persistent local storage
- **Document Loading:** PyPDF (with error handling)
- **Embeddings:**
  - Gemini Mode: `models/text-embedding-004` (Google's latest)
  - Llama Mode: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **LLMs:**
  - Google Gemini 3 Flash Preview (via `google.genai` SDK)
  - Meta Llama 3 (via Ollama)

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Rashad1019/rag_document.git
cd rag_document
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama (Llama Mode Only)
- Download from [ollama.com](https://ollama.com)
- Pull the model:
  ```bash
  ollama pull llama3
  ```

---

## 🔑 Configuration

### Gemini Setup (main.py)

1. **Create Environment File**
   ```bash
   cp .env.example .env
   ```

2. **Add Your Google API Key**
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```
   
   Get your API key from the [Google AI Studio](https://aistudio.google.com/app/apikey)

3. **Add Documents**
   Place your PDF files into the `data/` folder. The system will automatically:
   - Create the folder if it doesn't exist
   - Scan and ingest PDFs on first run
   - Allow incremental updates via the `update` command

---

## 🏃 Usage

### Option A: Gemini Mode (Recommended)
```bash
python main.py
```

**Features:**
- Automatic indexing on first run
- Type `update` to refresh when you add new PDFs
- Type `exit` or `quit` to stop the session
- Leverages Gemini 3's superior reasoning capabilities

### Option B: Llama 3 Mode (Privacy-First)
```bash
python main_llama.py
```

**Requirements:**
- Ensure Ollama is running in the background
- All processing happens locally on your machine

---

## 💡 Usage Examples

```
Gemini 3 RAG System Ready.
Commands: 'exit' to quit, 'update' to re-scan PDFs.

Your Question: What is RAG?
Reasoning with Gemini 3 Flash...

[Gemini 3 Flash]:
RAG stands for Retrieval-Augmented Generation. It connects the 
powerful reasoning of an LLM with unique information from your 
own documents to provide accurate, context-aware answers.

Your Question: update
Rebuilding database to ensure all files are current...
Scanning for new documents...
Processing 15 document pages...
Added 87 new chunks to the database.

Your Question: exit
```

---

## 🌟 What Makes This Different

This project elevates the standard RAG tutorial into a production-ready application:

### 1. **Auto-Ingestion Pipeline**
The system automatically scans the `data/` folder on startup. New PDFs are processed immediately without manual intervention.

### 2. **Persistent Vector Storage**
Unlike simple scripts that rebuild the database on every run, The Agency saves your vector index to disk. You only pay the indexing cost once.

### 3. **Dynamic Updates**
Added an `update` command inside the chat loop. Drop a new PDF while the script is running, type `update`, and the system re-indexes without restarting.

### 4. **Smart Rebuild Strategy**
The update mechanism performs a complete rebuild by:
- Deleting the existing collection
- Re-instantiating the vector store
- Processing all documents fresh to avoid "collection not initialized" errors

### 5. **Dual-Engine Architecture**
Separated logic into `main.py` (Cloud/Gemini) and `main_llama.py` (Local/Llama) so you can switch strategies based on your data privacy needs.

### 6. **Production Error Handling**
- API error catching with descriptive messages
- File loading errors don't crash the pipeline
- Environment variable validation
- Graceful handling of empty document folders

---

## 🛡️ Security & Privacy

- **`.gitignore` Protection:** Prevents accidental commits of API keys (`.env`) and vector databases (`chroma_db/`)
- **Local-First Option:** Llama mode keeps all data on your machine
- **No Data Retention:** Gemini API processes queries in real-time without storing your documents

**⚠️ Important:** Always keep your `GOOGLE_API_KEY` private. Never commit `.env` files to version control.

---

## 📚 How It Works

### The RAG Pipeline

1. **Document Loading:** PyPDFLoader extracts text from PDFs in the `data/` folder
2. **Chunking:** RecursiveCharacterTextSplitter breaks documents into 1000-character chunks with 200-character overlap
3. **Embedding:** Text chunks are converted to numerical vectors using Google's `text-embedding-004`
4. **Storage:** Vectors are stored in ChromaDB for fast similarity search
5. **Retrieval:** When you ask a question, the system finds the top 5 most relevant chunks
6. **Generation:** Context is passed to Gemini 3 Flash with strict instructions to answer only from the provided context

### Why This Architecture?

- **Chunk Overlap:** Ensures context isn't lost at chunk boundaries
- **k=5 Retrieval:** Balances context richness with token efficiency
- **Temperature 0.3:** Prioritizes factual accuracy over creative responses
- **System Instructions:** Prevents hallucination by constraining responses to source material

---

## 🔄 Future Enhancements

- [ ] Support for additional document formats (DOCX, TXT, Markdown)
- [ ] Duplicate detection to avoid re-indexing unchanged files
- [ ] Conversation history for multi-turn dialogue
- [ ] Web UI using Streamlit or Gradio
- [ ] Metadata filtering (search by document, date, tags)
- [ ] Hybrid search combining semantic and keyword matching

---

## 🙏 Acknowledgments

This project was inspired by and builds upon the tutorial **"Building a Multi-Document RAG System"** by [Aman Kharwal](https://thecleverprogrammer.com/2026/01/06/building-a-multi-document-rag-system/). I'm grateful for his clear explanation of RAG fundamentals, which provided the foundation for these enhancements.

**Original Tutorial:** [Building a Multi-Document RAG System | Aman Kharwal](https://thecleverprogrammer.com/2026/01/06/building-a-multi-document-rag-system/)

---

## 📬 Contact

**Rashad**  
📧 [Rashad19@outlook.com](mailto:Rashad19@outlook.com)  
🐙 [GitHub: Rashad1019](https://github.com/Rashad1019)

Questions, feedback, or collaboration ideas? Feel free to reach out!

---

## 📄 License

This project is open source and available under the MIT License. See `LICENSE` for more details.

---

## 🚀 Get Started Now

```bash
# Quick Start (Gemini Mode)
git clone https://github.com/Rashad1019/rag_document.git
cd rag_document
pip install -r requirements.txt
# Add your GOOGLE_API_KEY to .env
# Drop PDFs into data/
python main.py
```

**Happy Document Chatting! 🎉**