import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

import warnings
# Suppress the "Accessing __path__" and version warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning) 
# Also suppress any potential DeprecationWarnings from LangChain
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Simplified LangSmith Tracking
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    # Pass parameters directly to the model constructor
    llm = ChatOpenAI(
        model=engine, 
        openai_api_key=api_key, 
        temperature=temperature, 
        max_tokens=max_tokens
    )
    
    output_parser = StrOutputParser()
    
    # Using LCEL (LangChain Expression Language)
    chain = prompt | llm | output_parser
    
    answer = chain.invoke({'question': question})
    return answer

## UI Logic
st.title("Enhanced Q&A Chatbot With OpenAI")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

engine = st.sidebar.selectbox("Select OpenAI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=5000, value=150)

st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    if api_key:
        try:
            response = generate_response(user_input, api_key, engine, temperature, max_tokens)
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter the OpenAI API Key in the sidebar to proceed.")