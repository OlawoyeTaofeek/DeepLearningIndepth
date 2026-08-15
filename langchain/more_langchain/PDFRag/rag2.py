import streamlit as st
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
import os
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Asistant", layout="wide")

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")  # saves to folder next to your script
    collection = client.get_or_create_collection("vector_store")

    if collection.count() == 0:
        with st.spinner("Embedding documents..."):
            loader = PyPDFDirectoryLoader(path="research")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = splitter.split_documents(docs)

            embedding = get_embedding_model()
            texts = [s.page_content for s in splits]

            collection.upsert(
                ids=[str(i) for i in range(len(splits))],
                documents=texts,
                embeddings=embedding.embed_documents(texts),
                metadatas=[s.metadata for s in splits]
            )

    return collection

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        streaming=True
    )

def rag_query(user_input: str):
    collection = get_collection()
    embedding_model = get_embedding_model()
    llm = get_llm()

    # 1. Embed the user query
    query_embedding = embedding_model.embed_query(user_input)
    # 2. Retrieve top-k relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Build context string
    context = "\n\n".join([
        f"[Source: {m.get('source', 'unknown')}, Page {m.get('page', '?')}]\n{chunk}"
        for chunk, m in zip(chunks, metadatas)
    ])

    # 4. Call LLM with context
    messages = [
        SystemMessage(content=f"""You are a research assistant. 
        Answer the user's question using ONLY the context below.
        If the answer isn't in the context, say so clearly.

        Context:
        {context}"""),
        HumanMessage(content=user_input)
    ]

    return llm.stream(messages), chunks, metadatas, distances

# ─── UI ────────────────────────────────────────────────────
st.title("📄 Research Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources used"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src['source']} — Page {src['page']} | Similarity: {src['score']}")
                    st.markdown(f"> {src['chunk']}")

# Chat input
if user_input := st.chat_input("Ask something about your documents..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run RAG and stream response
    with st.chat_message("assistant"):
        stream, chunks, metadatas, distances = rag_query(user_input)
        response = st.write_stream(stream)

        # Show sources
        sources = [
            {
                "source": m.get("source", "unknown").split("/")[-1],  # just filename
                "page": m.get("page", "?"),
                "chunk": chunk,
                "score": f"{(1 - dist):.0%}"  # convert distance to similarity %
            }
            for chunk, m, dist in zip(chunks, metadatas, distances)
        ]

        with st.expander("📚 Sources used"):
            for src in sources:
                st.caption(f"📄 {src['source']} — Page {src['page']} | Similarity: {src['score']}")
                st.markdown(f"> {src['chunk']}")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })