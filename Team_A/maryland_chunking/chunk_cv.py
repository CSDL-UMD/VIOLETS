# NOT FINISHED
# after loading, chunk each url in the output spreadsheet


import os
from langchain.chat_models import init_chat_model

from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_pinecone import PineconeVectorStore
from google.colab import drive

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Load the spreadsheet
df = pd.read_excel(r'C:\Users\megan\Downloads\report_fixed.xlsx') #currently my personal path 
urls = df['url'].tolist()

chunk_list = []
for url in urls:
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')


    # Set up recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=210,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", "", "."] #update to prevent strange cutoffs, structureware splitting -> regex

    recursive_docs = text_splitter.split_text(soup.get_text())
    chunk_list.append(recursive_docs)
)

os.environ["OPENAI_API_KEY"] = "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"
from langchain_text_splitters import RecursiveCharacterTextSplitter


# [?] Google drive
drive.mount('/content/drive')

# 1. Embedding with cache
underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = LocalFileStore("/content/drive/MyDrive/embedding_cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="text-embedding-3-small"
)

# 2. Pinecone for vector storage
vectorstore = PineconeVectorStore.from_documents(
    recursive_docs,
    cached_embedder,  # Uses cache when embedding
    index_name="your-index-name"
)

if __name__ == "__main__":
    main()
''' 
viewing chunk code

print(f"Total # of chunks: {len(recursive_docs)}\n")
for i, doc in enumerate(recursive_docs):
    print(f"--- Chunk {i+1} ({len(doc)} characters) ---")
    print(doc)
    print()
    '''