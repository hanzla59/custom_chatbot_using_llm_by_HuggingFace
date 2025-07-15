# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.document_loaders import TextLoader
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.vectorstores import FAISS
# # from langchain.llms import Ollama
# from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA

# import os

# # Load your data
# loader = TextLoader("data/business_info.txt", encoding="utf-8")
# documents = loader.load()

# # Split into smaller chunks
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = splitter.split_documents(documents)

# # Embed using HuggingFace
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Create vector database
# vectorstore = FAISS.from_documents(docs, embedding_model)

# # Set up retriever
# retriever = vectorstore.as_retriever()

# # Use local model from Ollama
# # llm = Ollama(model="mistral")
# llm = OllamaLLM(model="mistral")

# # Create QA system
# qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# # Chat loop
# print("🤖 Chatbot is ready! Ask a question.")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ['exit', 'quit']:
#         print("Goodbye!")
#         break
#     # response = qa.run(user_input)
#     response = qa.invoke(user_input)
#     print("Bot:", response)





# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_ollama import OllamaLLM

# # Load your data
# loader = TextLoader("data/business_info.txt", encoding="utf-8")
# documents = loader.load()

# # Split into chunks
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = splitter.split_documents(documents)

# # Embeddings and vector DB
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(docs, embedding_model)
# retriever = vectorstore.as_retriever()

# # Correct Ollama class
# llm = OllamaLLM(model="mistral")
# qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# # Chat loop
# print("🤖 Chatbot is ready! Ask a question.")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ['exit', 'quit']:
#         print("Goodbye!")
#         break
#     response = qa.invoke(user_input)  # updated from .run()
#     print("Bot:", response)














# import os
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_ollama import OllamaLLM

# DB_PATH = "vectorstore"

# # Embeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Load or create vectorstore
# if os.path.exists(DB_PATH):
#     print("🔄 Loading existing vectorstore...")
#     vectorstore = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
# else:
#     print("📄 Creating vectorstore from text...")
#     loader = TextLoader("data/business_info.txt", encoding="utf-8")
#     documents = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.split_documents(documents)

#     vectorstore = FAISS.from_documents(docs, embedding_model)
#     vectorstore.save_local(DB_PATH)
#     print("✅ Vectorstore saved.")

# retriever = vectorstore.as_retriever()

# # Load LLM (locally with Ollama)
# llm = OllamaLLM(model="mistral")
# qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# # Chat loop
# print("🤖 Chatbot is ready! Ask a question.")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ['exit', 'quit']:
#         print("Goodbye!")
#         break
#     response = qa.invoke(user_input)
#     print("Bot:", response)













import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import time
from functools import lru_cache

DB_PATH = "vectorstore"
DATA_FILE = "data/business_info.txt"
VECTORSTORE_MTIME_FILE = "vectorstore/mtime.txt"  # Store last modification time

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Cache embedding generation
@lru_cache(maxsize=1000)
def get_query_embedding(query):
    return embedding_model.embed_query(query)

# Check if data file has changed
def data_file_changed():
    if not os.path.exists(DATA_FILE):
        return True
    if not os.path.exists(VECTORSTORE_MTIME_FILE):
        return True
    with open(VECTORSTORE_MTIME_FILE, "r") as f:
        last_mtime = float(f.read().strip())
    current_mtime = os.path.getmtime(DATA_FILE)
    return current_mtime > last_mtime

# Load or create vectorstore
if os.path.exists(DB_PATH) and not data_file_changed():
    print("🔄 Loading existing vectorstore...")
    vectorstore = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    print("📄 Creating vectorstore from text...")
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(DB_PATH)
    # Save modification time
    os.makedirs(os.path.dirname(VECTORSTORE_MTIME_FILE), exist_ok=True)
    with open(VECTORSTORE_MTIME_FILE, "w") as f:
        f.write(str(os.path.getmtime(DATA_FILE)))
    print("✅ Vectorstore saved.")

retriever = vectorstore.as_retriever()
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load LLM
llm = OllamaLLM(model="mistral") 
# llm = OllamaLLM(model="tinyllama") # Consider "tinyllama" for faster inference
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Chat loop
print("🤖 Chatbot is ready! Ask a question.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    start_time = time.time()
    response = qa.invoke(user_input)
    print("Bot:", response)
    print(f"Response time: {time.time() - start_time:.2f} seconds")