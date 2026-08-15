# ── Imports ──────────────────────────────────────────────────────────────────
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import tempfile
import os

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(find_dotenv(), override=True)

os.environ['HUGGING_FACE_ACCESS_TOKEN'] = os.getenv("HUGGING_FACE_ACCESS_TOKEN", "")

# Load embeddings once at startup — expensive to reload on every rerun
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Conversational RAG with PDF Uploads & Chat History")
st.write("Upload PDFs and chat with their content")

# ── API Key ───────────────────────────────────────────────────────────────────
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password")

if not api_key:
    st.info("Please enter your Groq API key in the sidebar to get started.")
    st.stop()   # Halt execution — nothing below runs without a key

# ── LLM ──────────────────────────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",   # Fixed: was empty string
)

# ── Session Management ────────────────────────────────────────────────────────
session_id = st.sidebar.text_input("🗂 Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}
    
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# ── PDF Upload & Processing ───────────────────────────────────────────────────
uploaded_files = st.sidebar.file_uploader(
    "📄 Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
)

@st.cache_resource(show_spinner="Indexing documents…")
def process_documents(_uploaded_files):
    """
    Cache the vector store so it isn't rebuilt on every Streamlit rerun.
    The leading underscore in _uploaded_files tells Streamlit not to
    hash this argument (UploadedFile objects aren't hashable).
    """
    documents = []

    for file in _uploaded_files:
        # Fixed: write to a proper temp file, then pass the PATH to PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)          
        documents.extend(loader.load())
        os.unlink(tmp_path)                     

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
    )
    chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store

if not uploaded_files:
    st.info("Upload one or more PDFs in the sidebar to begin.")
    st.stop()

vector_store = process_documents(tuple(uploaded_files))   # tuple so it's hashable for cache
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# ── Prompts ───────────────────────────────────────────────────────────────────

# Prompt 1: Rewrites the user question as a standalone question using chat history
contextualize_system_prompt = """
Given a chat history and the latest user question which might reference 
context in the chat history, formulate a standalone question that can be 
understood without the chat history. Do NOT answer the question — just 
reformulate it if needed, otherwise return it as is.
"""

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),  
    ("human", "{input}"),                              
])

# Prompt 2: Answers the question using retrieved context
qa_system_prompt = """
You are an assistant for question-answering tasks. Use the following pieces 
of retrieved context to answer the question. If you don't know the answer, 
say that you don't know. Keep the answer deep and well explained.

Context:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),  
    ("human", "{input}"),                               
])

# ── Build the RAG Chain ───────────────────────────────────────────────────────

# Step 1: Retriever that rewrites the question using history before searching
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,
)

# Step 2: Chain that stuffs retrieved docs into {context} and calls the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step 3: Full pipeline — retrieve then answer
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Step 4: Wrap with automatic history management
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",           
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ── Chat UI ───────────────────────────────────────────────────────────────────

# Render existing messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about your documents…")

if user_input:
    # Show user message immediately
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = conversational_rag_chain.invoke(
                {"input": user_input},          # Fixed: proper dict with "input" key
                config={
                    "configurable": {"session_id": session_id}
                }
            )

        answer = response["answer"]
        st.markdown(answer)                    

        # Show retrieved context chunks so you can verify the sources
        with st.expander("📎 View retrieved context chunks"):
            for i, doc in enumerate(response.get("context", []), 1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i}** — `{source}` · page {page}")
                st.text(doc.page_content)
                st.divider()

    # Save assistant reply to display history
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# ── Sidebar: Session Info ─────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### 🗃 Session History")
    history = get_session_history(session_id)

    if history.messages:
        for msg in history.messages:
            role = "🧑 You" if msg.type == "human" else "🤖 Assistant"
            st.markdown(f"**{role}:** {msg.content[:120]}{'…' if len(msg.content) > 120 else ''}")
    else:
        st.caption("No history yet for this session.")

    if st.button("🗑 Clear session history"):
        st.session_state.store[session_id] = ChatMessageHistory()
        st.session_state.chat_messages = []
        st.rerun()