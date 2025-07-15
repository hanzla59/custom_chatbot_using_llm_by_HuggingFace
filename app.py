# from flask import Flask, request, jsonify
# from langchain.chains import RetrievalQA
# from langchain_ollama import OllamaLLM
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# app = Flask(__name__)

# # Load vectorstore and model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
# retriever = vectorstore.as_retriever()
# llm = OllamaLLM(model="mistral")
# qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# @app.route("/ask", methods=["POST"])
# def ask():
#     data = request.json
#     question = data.get("question", "")
#     if not question:
#         return jsonify({"error": "No question provided"}), 400
#     answer = qa.invoke(question)
#     return jsonify({"answer": answer})

# if __name__ == "__main__":
#     app.run(port=5000)












import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from functools import lru_cache
import time

app = Flask(__name__)

DB_PATH = "vectorstore"
DATA_FILE = "data/business_info.txt"
VECTORSTORE_MTIME_FILE = "vectorstore/mtime.txt"

# Embeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")


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

# Load or create vectorstore at startup
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
    os.makedirs(os.path.dirname(VECTORSTORE_MTIME_FILE), exist_ok=True)
    with open(VECTORSTORE_MTIME_FILE, "w") as f:
        f.write(str(os.path.getmtime(DATA_FILE)))
    print("✅ Vectorstore saved.")

# retriever = vectorstore.as_retriever()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load LLM at startup
# llm = OllamaLLM(model="mistral")  
llm = OllamaLLM(model="tinyllama") # Consider "tinyllama" for faster inference

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# @app.route("/ask", methods=["POST"])
# def ask():
#     data = request.json
#     question = data.get("question", "")
#     if not question:
#         return jsonify({"error": "No question provided"}), 400
#     start_time = time.time()
#     answer = qa.invoke(question)
#     response_time = time.time() - start_time
#     return jsonify({"answer": answer, "response_time": response_time})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    start_time = time.time()
    # Embedding generation
    embed_start = time.time()
    _ = get_query_embedding(question)  # Trigger cached embedding
    embed_time = time.time() - embed_start
    
    # Retrieval
    retrieval_start = time.time()
    docs = retriever.get_relevant_documents(question)
    retrieval_time = time.time() - retrieval_start
    
    # LLM inference
    llm_start = time.time()
    answer = qa.invoke(question)
    llm_time = time.time() - llm_start
    
    total_time = time.time() - start_time
    return jsonify({
        "answer": answer,
        "total_time": total_time,
        "embed_time": embed_time,
        "retrieval_time": retrieval_time,
        "llm_time": llm_time
    })

if __name__ == "__main__":
    app.run(port=5000)