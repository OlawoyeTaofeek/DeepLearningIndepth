import os
import tempfile 
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is a sample text file.\nThis file is used to explain document loader usin TextLoader")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        docs = loader.load()
        for doc in docs:
            print("Document content:")
            print(doc)
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)

def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents from {pdf_path}")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1} Content Preview: {doc.page_content[:150]}")
        print(f"Metadata: {doc.metadata}")
        print("\n")

if __name__ == "__main__":
    # load_text_file()
    pdf_loader("langchain_demo.pdf")