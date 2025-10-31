import os
from flask import Flask, request, render_template_string, session, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain, create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Community/Cloud Imports
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.chat_models.huggingface import ChatHuggingFace
# from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import pinecone
from pinecone import ServerlessSpec

load_dotenv()

app = Flask(__name__)
app.secret_key = "super_secret_key"


# ENV variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
INDEX_NAME = "handson-chatbot-index"

# ---- HTML UI ----
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f3f4f7;
            margin: 0;
            padding: 0;
        }
        .container {
            display: flex;
            justify-content: center;
            align-items: stretch;  /* Stretch children to equal height */
            width: 100vw;
            height: 100vh;         /* or auto if you want natural height */
            box-sizing: border-box;
            padding: 40px 0;
        }

        .sidebar, .chatbox {
            height: 100%;         /* Fill container height equally */
        }

        .container {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            width: 100vw;
            height: 100vh;
            box-sizing: border-box;
            padding: 40px 0;
        }
        .sidebar {
            background: #fff;
            border-radius: 12px 0 0 12px;
            box-shadow: 2px 0 8px rgba(0,0,0,0.05);
            padding: 32px 24px;
            min-width: 320px;
            max-width: 340px;
            width: 28vw;
            box-sizing: border-box;
            border-right: 1px solid #e8e8e8;
        }
        .chatbox {
            background: #fff;
            border-radius: 0 12px 12px 0;
            box-shadow: 2px 0 10px rgba(0,0,0,0.06);
            padding: 32px 24px;
            min-width: 420px;
            max-width: 700px;
            width: 60vw;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
        }
        .chatbox h2 {
            margin-top: 0;
            color: #2563eb;
            margin-bottom: 16px;
        }
        .chat-messages {
            flex: 1 1 auto;
            max-height: 480px;
            min-height: 320px;
            overflow-y: auto;
            margin-bottom: 12px;
            padding-right: 8px;
            border-radius: 6px;
            border: 1px solid #eef1f5;
            background: #f9fbfc;
        }
        .message {
            margin-bottom: 18px;
            padding: 8px 14px;
            border-radius: 8px;
            background: #f7fafc;
        }
        .message.user {
            background: #e0e7ff;
            text-align: right;
        }
        .message.ai {
            background: #f1f5f9;
            text-align: left;
        }
        .sidebar input[type="text"], .sidebar input[type="file"] {
            width: 94%;
            margin: 8px 0 18px 0;
            padding: 7px 12px;
            border-radius: 4px;
            border: 1px solid #d1d5db;
            font-size: 15px;
        }
        .sidebar input[type="checkbox"] {
            margin-right: 8px;
        }
        .sidebar select {
            width: 80%;
            padding: 6px 8px;
            margin-bottom: 16px;
            font-size: 15px;
            border-radius: 3px;
            border: 1px solid #d1d5db;
        }
        .sidebar label {
            font-weight: bold;
            margin-top: 10px;
            display: block;
        }
        .sidebar input[type="submit"] {
            background: #2563eb;
            color: #fff;
            padding: 9px 18px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 18px;
            font-size: 16px;
            transition: 0.2s background;
        }
        .sidebar input[type="submit"]:hover {
            background: #1d4ed8;
        }

    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <form method="post" enctype="multipart/form-data">
                <label for="model">Choose Model:</label>
                <select name="model" id="model">
                    <option value="azure">Azure OpenAI</option>
                    <option value="huggingface">Hugging Face</option>
                </select>
                <label for="system_prompt">Custom System Prompt:</label>
                <input name="system_prompt" id="system_prompt" placeholder="(leave blank for default)" type="text">
                <label>
                    <input type="checkbox" name="pinecone_enabled" value="yes">
                    Enable Pinecone Retrieval
                </label>
                <label for="pdf_file">Upload PDF:</label>
                <input type="file" name="pdf_file" id="pdf_file" accept=".pdf">
                <label for="user_input">Your Message:</label>
                <input name="user_input" id="user_input" placeholder="Type your message here..." required type="text">
                <input type="submit" value="Send">
            </form>
        </div>
        <div class="chatbox">
            <h2>Conversation</h2>
            <div class="chat-messages" id="chat-messages">
                {% for msg in chat_history %}
                    <div class="message {{ 'user' if msg['role'] == 'User' else 'ai' }}">
                        <b>{{msg['role']}}:</b>
                        {{msg['text']}}
                    </div>
                {% endfor %}
            </div>
        </div>
    </div>
    <script>
        // Scroll to the bottom of the chat-messages div after load
        window.onload = function() {
            var chatDiv = document.getElementById('chat-messages');
            if(chatDiv){
                chatDiv.scrollTop = chatDiv.scrollHeight;
            }
        };
    </script>
</body>
</html>
"""

# -------- Helpers --------
def get_llm(model_choice):
    if model_choice == "azure":

        return AzureChatOpenAI(
            openai_api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version="2023-05-15",
            deployment_name=DEPLOYMENT_NAME,
            openai_api_type="azure",
            temperature=0.2,
        )
    elif model_choice == "huggingface":
        # 1. Import necessary classes
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        llm_base = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",#meta-llama/Llama-3.1-8B-Instruct
            huggingfacehub_api_token=HF_API_KEY,
            max_new_tokens=512,
            temperature=0.2,
        )
        chat_model = ChatHuggingFace(llm=llm_base)

        return chat_model

def setup_pinecone_index():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [x.name for x in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)

def store_pdf_in_pinecone(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    docs = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_KEY,
        openai_api_version="2023-05-15",
        openai_api_type="azure",
        chunk_size=2048,
    )
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    return vectorstore

@app.route("/", methods=["GET", "POST"])
def chat():
    session.setdefault("chat_history", [])
    session.setdefault("pdf_uploaded", False)
    memory = ConversationBufferMemory(return_messages=True)

    # Defaults
    llm = None
    vectorstore = None
    retriever = None

    user_prompt = ""
    response_text = ""

    if request.method == "POST":
        # Model selection & prompt
        model_choice = request.form.get("model")
        print(f"\n\n\n\n\n\nModel choice: {model_choice}\n\n\n\n\n\n")
        user_prompt = request.form.get("system_prompt", "")
        use_pinecone = request.form.get("pinecone_enabled") == "yes"
        user_msg = request.form.get("user_input", "")

        # Get or set system prompt
        default_system = (
            "You are a helpful and knowledgeable AI assistant. "
            "Your goal is to answer questions, provide information, and assist the user in any topic they request. "
            "If you do not know an answer, state clearly that you are unsure and offer to help with related information. "
        )

        # Get selected LLM
        llm = get_llm(model_choice)

        # Pinecone index setup only if enabled
        if use_pinecone:
            default_system = '''
                You are a helpful AI assistant. 
                Answer ONLY using information provided in the retrieved context below. 
                If the context does not help answer the user's question, reply: "Sorry, I can't answer that from the uploaded content."
            '''
            index = setup_pinecone_index()
            # If PDF uploaded (once per session)
            if "pdf_file" in request.files and request.files["pdf_file"].filename != "":
                pdf_file = request.files["pdf_file"]
                filepath = secure_filename(pdf_file.filename)
                pdf_file.save(filepath)
                vectorstore = store_pdf_in_pinecone(filepath)
                session["pdf_uploaded"] = True
            # Use existing vectorstore if PDF uploaded previously in session
            elif session.get("pdf_uploaded", False):
                vectorstore = PineconeVectorStore.from_existing_index(
                    embedding=AzureOpenAIEmbeddings(
                        deployment="text-embedding-3-small",
                        model="text-embedding-3-small",
                        azure_endpoint=AZURE_OPENAI_ENDPOINT,
                        openai_api_key=AZURE_OPENAI_KEY,
                        openai_api_version="2023-05-15",
                        openai_api_type="azure",
                        chunk_size=2048,
                    ),
                    index_name=INDEX_NAME,
                )
        # --- Building chains ---
        session["chat_history"].append({"role": "User", "text": user_msg})
        system_message = user_prompt if user_prompt.strip() else default_system
        if use_pinecone and vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            contextualize_question_prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("system", "Given the above conversation, create a search query to find relevant information."),
            ])
            history_aware_retriever = create_history_aware_retriever(
                llm=llm,
                retriever=retriever,
                prompt=contextualize_question_prompt,
            )
            retrieval_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "Based on the following context:\n{context}\n\nAnswer the user's question:\n{input}"),
            ])
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)
            retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
            response = retrieval_chain.invoke(
                {
                    "input": user_msg,
                    "chat_history": memory.load_memory_variables({}).get("history", []),
                }
            )
            response_text = response.get("answer", "")
        else:
            # Simple chain (no Pinecone retrieval)
            conversation_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ]
            )
            chain = ConversationChain(llm=llm, memory=memory, prompt=conversation_prompt)
            response_text = chain.invoke({"input": user_msg})["response"]

        memory.save_context({"input": user_msg}, {"output": response_text})
        session["chat_history"].append({"role": "AI", "text": response_text})

    return render_template_string(
        HTML_PAGE,
        chat_history=session["chat_history"],
    )

if __name__ == "__main__":
    app.run(debug=True)

#Needs to be done
#streaming
#without refreshing the page,
#the model selected, enable disable pinecone, custom system prompt needs to be as it is without refreshing