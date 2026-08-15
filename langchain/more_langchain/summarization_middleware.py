from langchain_core.tools import tool
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents import create_agent
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv, find_dotenv
import time

load_dotenv(find_dotenv())

summary_prompt = """
Summarize the main thrust of this conversation. 
What have the human and the assistant discussed so far?
Focus on they facts and requests

<messages>
Messages to summarize:
{messages}
</messages>
"""

retriever = WikipediaRetriever(
    top_k_results=1,
    doc_content_chars_max=20_000,
)


from requests.exceptions import JSONDecodeError

@tool
def fetch_wikipedia_data(query: str) -> str:
    """Fetch content of wikipedia page from top hit of a query"""
    
    for attempt in range(3):  # retry
        try:
            results = retriever.invoke(query)
            
            if results:
                return results[0].page_content
            
            return "(No data found)"
        
        except JSONDecodeError:
            # Wikipedia returned bad response
            time.sleep(1)
        
        except Exception as e:
            # Catch everything else
            time.sleep(1)

    return "Wikipedia service unavailable"
    

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[fetch_wikipedia_data],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            summary_prompt=summary_prompt,
            # Trigger summarization when 70% of context is used
            trigger=("fraction", 0.7),
            # Keep the most recent 30% of message
            keep=("fraction", 0.3),
            # No additional trimming before summarization
            trim_tokens_to_summarize=None
        )
    ]
)

# messages = [
#     HumanMessage(content="Tell me about AI"),
#     HumanMessage(content="Explain deep learning"),
#     HumanMessage(content="What is reinforcement learning?"),
#     HumanMessage(content="Now summarize everything")
# ]

# response = agent.invoke({"messages": messages})

from fastapi import FastAPI
from langchain_core.messages import HumanMessage

app = FastAPI()

@app.get("/info")
def info():
    return {
        "name": "wiki-agent",
        "description": "Wikipedia agent with summarization middleware"
    }

@app.post("/invoke")
def invoke_agent(input: dict):
    response = agent.invoke({
        "messages": [HumanMessage(content=input["message"])]
    })
    return {"response": str(response)}