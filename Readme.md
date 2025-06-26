# RAG-based Chatbot POC using Gemini API on Streamlit

A **Proof-of-Concept (PoC)** for a **Retrieval-Augmented Generation (RAG)** chatbot using **Google's Gemini Flash Lite** model for generation and **Gemini Embedding 001** for document embeddings. Built with **Streamlit** for an interactive web interface.

## 🌟 Features

### 🗂️ Index Management
- **Multiple Indexes**: Create, select, and delete multiple named knowledge bases (indexes).
- **Document Upload**: Support for PDF and text files for each index.
- **Text Extraction**: Automatic text extraction from PDFs.
- **Smart Chunking**: Intelligent text chunking with sentence boundary detection.
- **Vector Embeddings**: Generate embeddings using Gemini Embedding 001.
- **FAISS Indexing**: Fast similarity search with FAISS for each index.
- **Persistent Storage**: Save and load individual indexes using pickle files.

### 💬 Chat Interface
- **Semantic Search**: Finds relevant document chunks using vector similarity.
- **Context-Aware Responses**: Generates answers using both retrieved document context and the ongoing conversation history.
- **Conversation Memory**: Understands follow-up questions by maintaining multi-turn conversation context.
- **Source References**: Shows which source documents were used to generate an answer.
- **Downloadable Sources**: Provides download links for the referenced documents directly in the chat.
- **Adjustable Retrieval**: Allows configuration of how many document chunks to retrieve (top-k).
- **Chat History**: Maintains and displays the full conversation during the session.

### 🔧 Additional Features
- **Error Handling**: Robust error handling with user-friendly messages
- **Progress Indicators**: Loading spinners and progress bars
- **Index Management**: Clear and rebuild indices
- **File Validation**: Validate file types and handle errors gracefully

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Copy `.env.template` to `.env`
3. Replace `your_gemini_api_key_here` with your actual API key

```bash
cp .env.template .env
# Edit .env file with your API key
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Step 1: Manage Indexes
1. Navigate to the **"📚 RAG Management"** section in the sidebar.
2. Create a new index by giving it a name and clicking **"Create Index"**.
3. If you have existing indexes, you can select one from the dropdown menu.
4. You can also delete the currently selected index.

### Step 2: Upload Documents
1. With an index selected, use the file uploader to add PDF or text files to it.
2. Click **"Process Documents"** to extract text, generate embeddings, and add them to the selected index.
3. You can view statistics for the selected index, such as the number of documents and sources.

### Step 3: Chat with Your Documents
1. Switch to the **"💬 Chat Interface"** section in the sidebar.
2. The chat will query against the index you last selected in the RAG Management section.
3. Type your question in the chat input.
4. Receive AI-generated answers based on your documents, with conversational context.
5. Expand the **"References"** section below an answer to see which documents were used and download them.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   RAG System     │    │  Google Gemini  │
│                 │    │                  │    │                 │
│ • File Upload   │───▶│ • Text Extract   │───▶│ • Embeddings    │
│ • Chat Input    │    │ • Chunking       │    │ • Generation    │
│ • Display       │◄───│ • FAISS Search   │◄───│ • API Calls     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Pickle Storage  │
                       │                  │
                       │ • FAISS Index    │
                       │ • Documents      │
                       │ • Metadata       │
                       └──────────────────┘
```

## 📁 Project Structure

```
rag-chatbot-poc/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.template         # Environment variables template
├── README.md             # This file
├── faiss_index.pkl       # FAISS index (created after first use)
└── documents.pkl         # Document store (created after first use)
```

## 🔧 Configuration Options

### Embedding Model
- **Model**: `models/embedding-001`
- **Dimension**: 768
- **Task Type**: `retrieval_document` / `retrieval_query`

### Generation Model
- **Model**: `gemini-1.5-flash`
- **Context Window**: Large context support
- **Response Format**: Text generation

### Chunking Parameters
- **Chunk Size**: 1000 characters (configurable)
- **Overlap**: 200 characters (configurable)
- **Boundary Detection**: Sentence-aware splitting

### FAISS Configuration
- **Index Type**: `IndexFlatIP` (Inner Product)
- **Similarity**: Cosine similarity (L2 normalized)
- **Storage**: Persistent pickle files

## 🐛 Error Handling

The application includes comprehensive error handling for:

- **API Errors**: Rate limiting, token limits, network issues
- **File Processing**: Invalid PDFs, encoding issues, empty files
- **Index Operations**: Corrupted indices, missing files
- **User Input**: Invalid queries, empty inputs

## 🚀 Performance Optimizations

- **Batch Processing**: Efficient embedding generation
- **Caching**: Session state management
- **Lazy Loading**: Load indices only when needed
- **Memory Management**: Optimized vector operations

## 📊 Monitoring & Debugging

The application provides:

- **Progress Indicators**: Real-time processing updates
- **Index Statistics**: Document count, source tracking
- **Source References**: Transparency in answer generation
- **Error Messages**: User-friendly error reporting

## 🔮 Future Enhancements

### Planned Features
- **Multiple Index Support**: Separate indices for different document types
- **Advanced Chunking**: Semantic chunking strategies
- **Conversation Memory**: Multi-turn conversation context
- **Document Preprocessing**: Better text cleaning and formatting
- **Export Features**: Download chat history and summaries

### Potential Improvements
- **Hybrid Search**: Combine semantic and keyword search
- **Reranking**: Improve retrieval quality with reranking models
- **Streaming Responses**: Real-time response generation
- **Authentication**: User management and access control

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Google Gemini API** for powerful AI capabilities
- **Streamlit** for the amazing web framework
- **FAISS** for efficient vector search
- **LangChain** community for RAG inspiration

## 📞 Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check the error logs in the Streamlit interface
- Verify your API key configuration
- Ensure all dependencies are installed correctly

---

**Happy Chatting! 🤖📚**