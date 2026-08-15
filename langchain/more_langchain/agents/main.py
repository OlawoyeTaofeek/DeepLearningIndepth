import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool, tool
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.callbacks import StreamlitCallbackHandler
import arxiv
import wikipedia
import tempfile
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchAgent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL CSS — only hide Streamlit chrome, nothing else
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 820px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION — skip agent for simple chat
# ─────────────────────────────────────────────────────────────────────────────
CHAT_PATTERNS = re.compile(
    r"^(hi|hello|hey|yo|sup|good\s(morning|afternoon|evening|night)|"
    r"how are you|what('s| is) up|thanks|thank you|cheers|bye|goodbye|"
    r"cool|great|nice|awesome|ok|okay|got it|makes sense|sure|alright|"
    r"help me|what can you do|who are you)[!?.,\s]*$",
    re.IGNORECASE,
)

RESEARCH_KEYWORDS = re.compile(
    r"\b(paper|papers|arxiv|research|study|studies|find|search|look up|"
    r"what is|explain|summarise|summarize|define|compare|"
    r"latest|recent|news|article|document|url|pdf|who|where|when|why|how)\b",
    re.IGNORECASE,
)

def needs_agent(prompt: str) -> bool:
    s = prompt.strip()
    if len(s) < 60 and CHAT_PATTERNS.match(s):
        return False
    if len(s) < 30 and not RESEARCH_KEYWORDS.search(s):
        return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────
@tool("arxiv_search")
def arxiv_search_tool(query: str) -> str:
    """Search academic research papers on arXiv. Use for AI, ML, physics,
    math, computer science, and any technical research paper questions."""
    try:
        client = arxiv.Client(page_size=3, delay_seconds=5.0, num_retries=3)
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for r in client.results(search):
            if "withdrawn" in r.title.lower():
                continue
            results.append(
                f"**{r.title}**\n"
                f"Published: {r.updated.date()} | Authors: {', '.join(a.name for a in r.authors[:4])}\n"
                f"Summary: {r.summary[:1200]}\nLink: {r.entry_id}"
            )
        return "\n\n---\n\n".join(results) if results else f"No arXiv papers found for: '{query}'"
    except Exception as e:
        if "429" in str(e) or "503" in str(e):
            return "arXiv is rate-limiting. Fall back to DuckDuckGo."
        return f"arXiv error: {e}"


def build_tools(openai_api_key: str, rag_retriever=None):
    wikipedia.set_user_agent("ResearchAgent/3.0 (research@example.com)")
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=4000)
    )
    ddg = DuckDuckGoSearchRun(
        api_wrapper=DuckDuckGoSearchAPIWrapper(region="en-us", max_results=5),
        name="DuckDuckGo Search",
    )
    tools = [arxiv_search_tool, wiki, ddg]
    if rag_retriever:
        rag_tool = create_retriever_tool(
            rag_retriever,
            name="document_search",
            description=(
                "Search the user's uploaded PDFs and URLs. "
                "Use this FIRST when the question might relate to uploaded content."
            ),
        )
        tools.insert(0, rag_tool)
    return tools


def build_rag_retriever(openai_api_key: str, sources: list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(sources)
    if not chunks:
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "messages": [
        {
            "role": "assistant",
            "content": (
                "👋 Hey! I'm **ResearchAgent** — part conversational assistant, part research engine.\n\n"
                "Chat with me normally, or ask me to search **arXiv**, **Wikipedia**, and the web. "
                "Upload PDFs or paste URLs in the sidebar to query your own documents.\n\n"
                "What would you like to explore?"
            ),
        }
    ],
    "rag_docs": [],
    "rag_retriever": None,
    "rag_source_names": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 ResearchAgent")
    st.caption("arXiv · Wikipedia · Web · Your Docs")
    st.divider()

    # — Settings —
    st.subheader("Settings")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Required for research queries and document indexing.",
    )
    openai_api_key = api_key_input or os.getenv("OPENAI_API_KEY", "")

    model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o-mini is fast and cheap; gpt-4o is more capable.",
    )
    st.divider()

    # — Active Tools —
    st.subheader("Active Tools")
    rag_active = st.session_state.rag_retriever is not None
    col1, col2 = st.columns(2)
    col1.success("arXiv")
    col2.success("Wikipedia")
    col3, col4 = st.columns(2)
    col3.success("DuckDuckGo")
    if rag_active:
        col4.success("Your Docs ✓")
    else:
        col4.info("Your Docs")
    st.divider()

    # — Knowledge Base —
    st.subheader("Knowledge Base")
    url_input = st.text_input("Paste a URL", placeholder="https://example.com/article")
    if st.button("Add URL", use_container_width=True):
        if not url_input.strip():
            st.warning("Enter a URL first.")
        elif not openai_api_key:
            st.error("Add your OpenAI API key first.")
        else:
            with st.spinner("Fetching and indexing..."):
                try:
                    docs = WebBaseLoader(url_input.strip()).load()
                    st.session_state.rag_docs.extend(docs)
                    st.session_state.rag_source_names.append(url_input.strip()[:60])
                    st.session_state.rag_retriever = build_rag_retriever(
                        openai_api_key, st.session_state.rag_docs
                    )
                    st.success(f"Added — {len(docs)} page(s) indexed.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        if not openai_api_key:
            st.error("Add your OpenAI API key first.")
        else:
            new_files = [f for f in uploaded if f.name not in st.session_state.rag_source_names]
            if new_files:
                with st.spinner(f"Indexing {len(new_files)} PDF(s)..."):
                    for uf in new_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name
                        try:
                            docs = PyPDFLoader(tmp_path).load()
                            st.session_state.rag_docs.extend(docs)
                            st.session_state.rag_source_names.append(uf.name)
                        finally:
                            os.unlink(tmp_path)
                    st.session_state.rag_retriever = build_rag_retriever(
                        openai_api_key, st.session_state.rag_docs
                    )
                    st.success(f"Indexed {len(new_files)} PDF(s).")

    if st.session_state.rag_source_names:
        st.info(
            f"**{len(st.session_state.rag_source_names)} source(s) loaded** — "
            f"{len(st.session_state.rag_docs)} chunks indexed\n\n"
            + "\n".join(f"• {n}" for n in st.session_state.rag_source_names[-5:])
        )
        if st.button("Clear Knowledge Base", use_container_width=True):
            st.session_state.rag_docs = []
            st.session_state.rag_retriever = None
            st.session_state.rag_source_names = []
            st.rerun()
    else:
        st.caption("No documents loaded yet. Paste a URL or upload PDFs above.")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. What would you like to explore?"}
        ]
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔬 ResearchAgent")
st.caption("arXiv · Wikipedia · Web Search · Your Documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────────────────────────────────────
# CHAT HANDLER
# ─────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything — research papers, topics, or your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── PATH 1: Conversational — no agent needed ──
    if not needs_agent(prompt):
        with st.chat_message("assistant"):
            if openai_api_key:
                llm_chat = ChatOpenAI(
                    openai_api_key=openai_api_key,
                    model_name=model_choice,
                    temperature=0.7,
                )
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                resp = llm_chat.invoke(history)
                response = resp.content
            else:
                lp = prompt.strip().lower()
                if re.match(r"hi|hello|hey|yo", lp):
                    response = "Hey there! 👋 Ask me anything — I can search research papers, Wikipedia, or the web."
                elif re.match(r"how are you", lp):
                    response = "Doing great, thanks! What can I help you research today?"
                elif re.match(r"thank|thanks|cheers", lp):
                    response = "You're welcome! Let me know if there's anything else."
                elif re.match(r"bye|goodbye", lp):
                    response = "Goodbye! Come back anytime. 👋"
                elif re.match(r"what can you do|who are you|help", lp):
                    response = (
                        "I'm **ResearchAgent**. I can:\n\n"
                        "- 📄 Search **arXiv** for academic papers\n"
                        "- 🌐 Look up **Wikipedia** for facts and concepts\n"
                        "- 🔍 Search the **web** for current info\n"
                        "- 📚 Query your **uploaded PDFs or URLs**\n\n"
                        "Add your OpenAI API key in the sidebar to get started!"
                    )
                else:
                    response = "Add your OpenAI API key in the sidebar to enable full research features."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # ── PATH 2: Research query — spin up agent ──
    else:
        if not openai_api_key:
            with st.chat_message("assistant"):
                st.warning("Please add your **OpenAI API key** in the sidebar to use research features.")
            st.stop()

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_choice,
            streaming=True,
            temperature=0.1,
        )
        tools = build_tools(openai_api_key, st.session_state.rag_retriever)
        rag_active = st.session_state.rag_retriever is not None

        agent_prefix = f"""You are ResearchAgent, a thorough and precise research assistant.

Available tools:
{'- document_search: Search uploaded documents/URLs — use this FIRST for any question about uploaded content.' if rag_active else ''}
- arxiv_search: Academic papers on arXiv (AI/ML, physics, math, CS).
- WikipediaQueryRun: General facts, history, people, concepts.
- DuckDuckGo Search: Current events, news, anything recent.

Rules:
1. {'Check document_search first when documents are loaded.' if rag_active else 'Use arxiv_search for academic/technical questions.'}
2. Use multiple tools when needed for a thorough answer.
3. Cite sources with links where available.
4. If arXiv rate-limits, fall back to DuckDuckGo immediately.
5. Respond in clear, well-structured markdown.
"""

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=7,
            early_stopping_method="generate",
            agent_kwargs={"prefix": agent_prefix},
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = agent.run(prompt, callbacks=[st_cb])
            except Exception as e:
                response = f"⚠️ Something went wrong: `{e}`\n\nPlease try rephrasing."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})