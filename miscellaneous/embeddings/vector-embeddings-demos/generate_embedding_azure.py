import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
openai_client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"] + "/openai/v1/",
    api_key=token_provider,
)
MODEL_NAME = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
MODEL_DIMENSIONS = int(os.environ["AZURE_OPENAI_EMBEDDING_DIMENSIONS"])

embeddings_response = openai_client.embeddings.create(
    model=MODEL_NAME,
    dimensions=MODEL_DIMENSIONS,
    input="hello world",
)
embedding = embeddings_response.data[0].embedding

print(len(embedding))
print(embedding)