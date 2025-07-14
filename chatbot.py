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



from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# Load your data
loader = TextLoader("data/business_info.txt", encoding="utf-8")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings and vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# Correct Ollama class
llm = OllamaLLM(model="mistral")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Chat loop
print("🤖 Chatbot is ready! Ask a question.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    response = qa.invoke(user_input)  # updated from .run()
    print("Bot:", response)
