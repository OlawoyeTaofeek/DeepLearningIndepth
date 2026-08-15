import streamlit as st 
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.callbacks import StreamlitCallbackHandler
import arxiv
import wikipedia
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Custom Arxiv Tool (Bypasses LangChain's broken max_results=100 issue)
@tool("arxiv_search")
def arxiv_search_tool(query: str) -> str:
    """Search academic research papers on arXiv. Use this for ANY question about 
    AI papers, deep learning research, machine learning models, transformers, 
    neural networks, scientific papers, or technical research. Input should be 
    a search query string."""
    try:
        # Define a respectful client instance with a built-in 5-second delay policy
        client = arxiv.Client(page_size=2, delay_seconds=5.0, num_retries=3)
        
        # Enforce max_results=2 here directly so the API query parameter is clean
        search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in client.results(search):
            paper_info = (
                f"Title: {result.title}\n"
                f"Published: {result.updated.date()}\n"
                f"Authors: {', '.join(a.name for a in result.authors)}\n"
                f"Summary: {result.summary[:1500]}\n"  # Keep characters short
                f"Link: {result.entry_id}\n"
                "---"
            )
            results.append(paper_info)
            
        if not results:
            return f"No papers found on arXiv matching the query: '{query}'."
            
        return "\n\n".join(results)

    except Exception as e:
        # Fallback gracefully if the user's IP is still soft-banned by arXiv
        if "429" in str(e) or "503" in str(e):
            return "⚠️ arXiv servers are rate-limiting requests right now. Please try your query again in a few seconds or use Wikipedia/DuckDuckGo Search instead."
        return f"An error occurred while scanning arXiv: {str(e)}"

# 2. Clean Wikipedia Setup
wikipedia.set_user_agent("MyDeepLearningProject/1.0 (your_email@example.com)")
api_wikipedia_wrapper = WikipediaAPIWrapper()
wiki_tool = WikipediaQueryRun(api_wrapper=api_wikipedia_wrapper)

# 3. Clean DuckDuckGo Setup
search_wrapper = DuckDuckGoSearchAPIWrapper(region="en-us", max_results=3)
search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper, name="DuckDuckGo Search")

# Aggregate tools list
tools = [arxiv_search_tool, wiki_tool, search_tool]

# ==========================================
# STREAMLIT UI SETUP
# ==========================================

st.set_page_config(page_title="LangChain - Chat with search", page_icon="🔎", layout="wide")
st.title("🔎 LangChain - Chat with search")

st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
""")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Fallback to look for an environment variable if input is empty
openai_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display past message history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Handle user interaction
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key in the sidebar to continue.")
        st.stop()

    # Initialize LLM
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini", streaming=True)
    
    # Re-instantiate the standard zero-shot ReAct agent safely
    search_agent = initialize_agent(
        tools=tools, 
        llm=llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        handle_parsing_errors=True,
    agent_kwargs={
        "prefix": """You are a helpful research assistant with access to three tools:
            - arxiv_search_tool: for ANY research papers, AI/ML topics, scientific queries
            - WikipediaQueryRun: for general knowledge, history, concepts, people
            - DuckDuckGo Search: for current news, recent events, web results

            IMPORTANT: For questions about papers, research, transformers, attention mechanism, 
            or any academic topic — ALWAYS use arxiv_search_tool first.
        """
        }
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Pass raw user string text directly
        response = search_agent.run(prompt, callbacks=[st_cb])
        
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)