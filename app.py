import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import pickle
import os
import tempfile
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import fitz  # PyMuPDF
from io import BytesIO
import re
from dotenv import load_dotenv
import os
from contextlib import redirect_stderr

from data_models import DocumentChunk

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chatbot POC",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


class RAGSystem:
    """RAG System class handling embeddings, indexing, and retrieval for multiple indexes"""

    def __init__(self, index_dir="indexes"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            st.error("Please set GEMINI_API_KEY in your .env file")
            st.stop()

        genai.configure(api_key=self.api_key)
        self.embedding_model = "models/embedding-001"
        self.generation_model = "gemini-2.5-flash-lite-preview-06-17"

        # Directory to store index files
        self.index_dir = index_dir
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        # In-memory cache for indexes
        self.indexes = {}
        self.load_all_indexes()

    def get_index_path(self, index_name: str) -> str:
        """Get the file path for a given index name"""
        return os.path.join(self.index_dir, f"{index_name}.pkl")

    def get_available_indexes(self) -> List[str]:
        """Get a list of all available (saved) index names"""
        return [f.replace(".pkl", "") for f in os.listdir(self.index_dir) if f.endswith(".pkl")]

    def create_index(self, index_name: str):
        """Create a new, empty index"""
        if index_name in self.indexes:
            st.warning(f"Index '{index_name}' already exists.")
            return
        self.indexes[index_name] = {"index": None, "documents": []}
        self.save_index(index_name)
        st.success(f"Index '{index_name}' created successfully.")

    def delete_index(self, index_name: str):
        """Delete an index from memory and disk"""
        if index_name in self.indexes:
            del self.indexes[index_name]
        
        index_path = self.get_index_path(index_name)
        if os.path.exists(index_path):
            os.remove(index_path)
            st.success(f"Index '{index_name}' deleted successfully.")
        else:
            st.warning(f"No saved file found for index '{index_name}'.")

    def load_index(self, index_name: str):
        """Load a specific index from disk into memory"""
        index_path = self.get_index_path(index_name)
        try:
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    data = pickle.load(f)
                    self.indexes[index_name] = {
                        "index": data.get("index"),
                        "documents": data.get("documents", []),
                    }
        except Exception as e:
            st.warning(f"Could not load index '{index_name}': {str(e)}")
            if index_name in self.indexes:
                del self.indexes[index_name]

    def load_all_indexes(self):
        """Load all indexes from the index directory"""
        for index_name in self.get_available_indexes():
            self.load_index(index_name)

    def save_index(self, index_name: str):
        """Save a specific index from memory to disk"""
        if index_name not in self.indexes:
            st.error(f"Index '{index_name}' not found in memory.")
            return

        index_path = self.get_index_path(index_name)
        try:
            with open(index_path, "wb") as f:
                pickle.dump(self.indexes[index_name], f)
        except Exception as e:
            st.error(f"Error saving index '{index_name}': {str(e)}")

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file using PyMuPDF, suppressing MuPDF errors."""
        try:
            file_bytes = pdf_file.read()
            # PyMuPDF can be noisy with non-critical errors, so we suppress them
            with open(os.devnull, 'w') as devnull:
                with redirect_stderr(devnull):
                    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                        text = "\n".join(page.get_text() for page in doc)
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        """Split text into overlapping chunks"""
        text = re.sub(r"\s+", " ", text.strip())
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                boundary = max(text.rfind(s, start, end) for s in (".", "?", "!"))
                if boundary > start + chunk_size - 200:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        progress_bar = st.progress(0)
        progress_text = st.empty()
        for i, text in enumerate(texts):
            try:
                progress_text.text(f"Generating embeddings... {i+1}/{len(texts)}")
                progress_bar.progress((i + 1) / len(texts))
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(np.array(result["embedding"]))
                time.sleep(0.1)
            except Exception as e:
                st.error(f"Error generating embedding for chunk {i+1}: {str(e)}")
                embeddings.append(np.zeros(768))
        progress_bar.empty()
        progress_text.empty()
        return embeddings

    def create_faiss_index(self, embeddings: List[np.ndarray]) -> faiss.Index:
        """Create FAISS index from embeddings"""
        if not embeddings:
            return None
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        embeddings_array = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        return index

    def add_documents(self, index_name: str, texts: List[str], source: str):
        """Add documents to a specific index"""
        if index_name not in self.indexes:
            st.error(f"Index '{index_name}' does not exist. Please create it first.")
            return
        if not texts:
            return

        all_chunks = []
        for text in texts:
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)
        if not all_chunks:
            st.warning("No text chunks generated from the documents")
            return

        embeddings = self.generate_embeddings(all_chunks)
        current_docs = self.indexes[index_name]["documents"]
        new_documents = [
            DocumentChunk(
                content=chunk,
                source=source,
                chunk_id=len(current_docs) + i,
                embedding=embedding,
            )
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings))
        ]

        current_docs.extend(new_documents)
        self.indexes[index_name]["documents"] = current_docs
        all_embeddings = [doc.embedding for doc in current_docs]
        self.indexes[index_name]["index"] = self.create_faiss_index(all_embeddings)
        self.save_index(index_name)
        st.success(f"Added {len(all_chunks)} chunks from {source} to index '{index_name}'")

    def search(self, index_name: str, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Search for relevant documents in a specific index"""
        if index_name not in self.indexes or not self.indexes[index_name].get("index"):
            return []

        index_data = self.indexes[index_name]
        faiss_index = index_data["index"]
        documents = index_data["documents"]
        try:
            result = genai.embed_content(
                model=self.embedding_model, content=query, task_type="retrieval_query"
            )
            query_embedding = np.array([result["embedding"]]).astype("float32")
            faiss.normalize_L2(query_embedding)
            scores, indices = faiss_index.search(query_embedding, top_k)
            return [documents[idx] for score, idx in zip(scores[0], indices[0]) if idx < len(documents)]
        except Exception as e:
            st.error(f"Error during search in index '{index_name}': {str(e)}")
            return []

    def generate_response(self, query: str, context_docs: List[DocumentChunk], chat_history: List[dict]) -> tuple[str, list[DocumentChunk]]:
        """Generate response using Gemini with retrieved context and conversation history."""
        context = "\n".join([doc.content for doc in context_docs])

        # Construct conversation history string
        history_str = ""
        for message in chat_history:
            history_str += f"Human: {message['question']}\n"
            history_str += f"Assistant: {message['answer']}\n"

        prompt = f"""You are a helpful assistant. Answer the user's current question based on the conversation history and the following context.
        If the answer is not available in the context, say 'I am sorry, I cannot answer this question based on the available information.'

        Conversation History:
        {history_str}

        Context:
        {context}

        Current Question: {query}
        """
        if not context_docs:
            prompt = f"""You are a helpful assistant. Answer the user's current question based on the conversation history.
            If you cannot answer, say 'I am sorry, I cannot answer this question based on the available information.'

            Conversation History:
            {history_str}

            Current Question: {query}
            """

        try:
            model = genai.GenerativeModel(self.generation_model)
            response = model.generate_content(prompt)
            return response.text, context_docs
        except Exception as e:
            return f"Error generating response: {str(e)}", []


# --- App State Management ---
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

rag_system = st.session_state.rag_system
available_indexes = rag_system.get_available_indexes()
if not st.session_state.selected_index and available_indexes:
    st.session_state.selected_index = available_indexes[0]

# --- UI Rendering ---
st.title("🤖 RAG-based Chatbot POC")
st.markdown("*Powered by Google Gemini Flash Lite and Embedding 001*")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    section = st.radio("Select Section:", ["📚 RAG Management", "💬 Chat Interface"], key="section")
    st.divider()

    if section == "📚 RAG Management":
        st.subheader("⚙️ Index Management")

        # Create new index
        new_index_name = st.text_input("New Index Name", placeholder="e.g., project-docs")
        if st.button("Create Index") and new_index_name:
            rag_system.create_index(new_index_name)
            st.session_state.selected_index = new_index_name
            st.rerun()

        st.divider()

        # Select and manage existing index
        if available_indexes:
            st.session_state.selected_index = st.selectbox(
                "Select Index",
                options=available_indexes,
                index=available_indexes.index(st.session_state.selected_index) if st.session_state.selected_index in available_indexes else 0,
            )

            if st.button("🗑️ Delete Selected Index", type="secondary"):
                rag_system.delete_index(st.session_state.selected_index)
                st.session_state.selected_index = None
                st.rerun()
        else:
            st.info("No indexes found. Create one to get started.")

        st.divider()
        
        # Document upload for selected index
        if st.session_state.selected_index:
            st.subheader(f"📂 Upload to '{st.session_state.selected_index}'")
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help=f"Upload files to the '{st.session_state.selected_index}' index",
            )
            if uploaded_files and st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        text = rag_system.extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else uploaded_file.read().decode("utf-8")
                        if text.strip():
                            rag_system.add_documents(st.session_state.selected_index, [text], uploaded_file.name)
                        else:
                            st.warning(f"No text found in {uploaded_file.name}")
                st.rerun()

    else:  # Chat Interface
        st.subheader("💬 Chat Settings")
        top_k = st.slider("Chunks to retrieve:", 1, 10, 5)
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Main content area
if section == "📚 RAG Management":
    st.header("📚 RAG Management")
    if st.session_state.selected_index:
        st.info(f"Managing index: **{st.session_state.selected_index}**")
        index_data = rag_system.indexes.get(st.session_state.selected_index, {})
        documents = index_data.get("documents", [])
        if documents:
            st.subheader("📊 Index Statistics")
            st.metric("Total Documents", len(documents))
            sources = list(set(doc.source for doc in documents))
            st.write("**Sources:**")
            for source in sources:
                count = sum(1 for doc in documents if doc.source == source)
                st.write(f"• {source}: {count} chunks")
        else:
            st.info("👈 Upload documents using the sidebar to populate this index.")
    else:
        st.info("👈 Create or select an index from the sidebar to begin.")

elif section == "💬 Chat Interface":
    st.header("💬 Chat Interface")
    if not st.session_state.selected_index:
        st.warning("⚠️ Please select a knowledge base (index) from the RAG Management section first.")
    else:
        st.info(f"Querying against index: **{st.session_state.selected_index}**")
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                st.markdown(f"**🙋‍♂️ You:** {message['question']}")
                st.markdown(f"**🤖 Assistant:** {message['answer']}")
                if message.get("sources"):
                    with st.expander("References"):
                        for i, doc in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}:** *{doc.source}* (Chunk {doc.chunk_id})")
                            st.info(doc.content)
                st.divider()

        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question:", key="user_input")
            submit_button = st.form_submit_button("Send", type="primary")

        if submit_button and user_question:
            with st.spinner("Searching and generating response..."):
                relevant_docs = rag_system.search(st.session_state.selected_index, user_question, top_k)
                response, sources = rag_system.generate_response(user_question, relevant_docs, st.session_state.chat_history)
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response,
                    "sources": sources
                })
                st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, Google Gemini API, and FAISS*")
