from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version

from langchain_openai import ChatOpenAI

print(f"Langchain core version: {version('langchain-core')}")
print(f"Langgraph version:      {version('langgraph')}")
print(f"Langchain OpenAI version: {version('langchain-openai')}")

def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say completed in 2 sentences making it sound wonderful")
    print(f"Response from OpenAI: {response.content}")
    print("Setup completed")

if __name__ == "__main__":
    main()