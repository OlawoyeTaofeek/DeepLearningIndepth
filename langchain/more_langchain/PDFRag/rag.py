from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
import os
import streamlit as st 


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Asistant", layout="wide")


@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        streaming=True
    )

@st.cache_resource
def get_vectorstore():
    embedding = get_embedding_model()
    
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embedding,
        persist_directory="./chroma_db"
    )
    
    # Only embed if empty
    if vectorstore._collection.count() == 0:
        with st.spinner("Embedding documents..."):
            loader = PyPDFDirectoryLoader(path="research")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = splitter.split_documents(docs)
            
            vectorstore.add_documents(splits)  # handles embedding internally
    
    return vectorstore

def rag_query(user_input: str):
    vectorstore = get_vectorstore()
    llm = get_llm()

    # Retrieve — returns Document objects with .page_content and .metadata
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(user_input)

    # Build context
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', '').split('/')[-1]}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])

    # Call LLM
    messages = [
        SystemMessage(content=f"""You are a research assistant.
        Answer using ONLY the context below. If the answer isn't there, say so.

        Context:
        {context}"""),
        HumanMessage(content=user_input)
    ]

    stream = llm.stream(messages)
    return stream, docs

if user_input := st.chat_input("Ask something about your documents..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        stream, source_docs = rag_query(user_input)
        response = st.write_stream(stream)

        with st.expander("📚 Sources used"):
            for doc in source_docs:
                st.caption(f"📄 {doc.metadata.get('source', '').split(chr(92))[-1]} — Page {doc.metadata.get('page', '?')}")
                st.markdown(f"> {doc.page_content[:300]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": source_docs
    })