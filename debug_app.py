#!/usr/bin/env python
import sys
import traceback
from dotenv import load_dotenv

print("=" * 50)
print("Starting Health Care ChatBot Debug")
print("=" * 50)

try:
    print("\n[1/10] Loading environment...")
    load_dotenv()
    print("✓ Environment loaded")
    
    print("\n[2/10] Importing Flask...")
    from flask import Flask, render_template, jsonify, request
    print("✓ Flask imported")
    
    print("\n[3/10] Importing embeddings helper...")
    from src.helper import download_hugging_face_embeddings
    print("✓ Helper imported")
    
    print("\n[4/10] Importing Pinecone...")
    from langchain_pinecone import PineconeVectorStore
    print("✓ Pinecone imported")
    
    print("\n[5/10] Importing LangChain components...")
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    print("✓ LangChain components imported")
    
    print("\n[6/10] Importing Google GenAI...")
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✓ Google GenAI imported")
    
    print("\n[7/10] Importing prompt...")
    from src.prompt import *
    print("✓ Prompt imported")
    
    import os
    
    print("\n[8/10] Creating Flask app...")
    app = Flask(__name__)
    print("✓ Flask app created")
    
    print("\n[9/10] Setting up API keys...")
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    
    if not PINECONE_API_KEY:
        print("⚠ WARNING: PINECONE_API_KEY not found in environment")
    else:
        print("✓ PINECONE_API_KEY found")
    
    if not GOOGLE_API_KEY:
        print("⚠ WARNING: GOOGLE_API_KEY not found in environment")
    else:
        print("✓ GOOGLE_API_KEY found")
    
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✓ Environment variables set")
    
    print("\n[10/10] Loading all components...")
    print("  - Downloading embeddings...")
    embeddings = download_hugging_face_embeddings()
    print("    ✓ Embeddings loaded")
    
    print("  - Connecting to Pinecone index 'quickstart'...")
    index_name = "quickstart"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print("    ✓ Pinecone connected")
    
    print("  - Setting up retriever...")
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("    ✓ Retriever ready")
    
    print("  - Initializing ChatGoogleGenerativeAI...")
    chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    print("    ✓ Chat model initialized")
    
    print("  - Setting up prompt template...")
    from src.prompt import system_prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    print("    ✓ Prompt template ready")
    
    print("  - Creating RAG chain...")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | chatModel
        | StrOutputParser()
    )
    print("    ✓ RAG chain created")
    
    print("\n" + "=" * 50)
    print("✓ ALL COMPONENTS INITIALIZED SUCCESSFULLY!")
    print("=" * 50)
    
    @app.route("/")
    def index():
        return render_template('chat.html')
    
    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        input = msg
        print(f"\n[Chat] User input: {input}")
        response = rag_chain.invoke(msg)
        print(f"[Chat] Response: {response}")
        return str(response)
    
    print("\nStarting Flask server on 0.0.0.0:8080...")
    print("-" * 50)
    app.run(host="0.0.0.0", port=8080, debug=True)

except Exception as e:
    print("\n❌ ERROR OCCURRED:")
    print("=" * 50)
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print("\nFull Traceback:")
    print("-" * 50)
    traceback.print_exc()
    print("-" * 50)
    print("\nDEBUGGING INFO:")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    sys.exit(1)
