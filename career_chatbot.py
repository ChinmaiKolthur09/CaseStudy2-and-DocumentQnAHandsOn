import os
import pandas as pd
from flask import Flask, request,render_template, render_template_string, session
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import ConversationChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

import pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import requests
import smtplib
from email.mime.text import MIMEText

# Load environment variables
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "career-assistant"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# Pinecone setup (semantic skill & role retrieval)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

pdf_filename = "CAREER_BOOKLET.pdf"
pdf_loader = PyPDFLoader(pdf_filename)
data = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=100
)
chunks = text_splitter.split_documents(data)

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

llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2023-05-15",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_type="azure",
    temperature=0.2,
)

# --- System prompt for career guidance ---
system_message_template = SystemMessagePromptTemplate.from_template(
    "You are a career guidance AI. Use conversation history to remember user's interests and goals. "
    "Answer user questions by grounding your responses on the provided career context. "
    "If the user gets off-topic, gently redirect to career guidance. "
    "Ask open-ended and clarifying questions to help the user discover their strengths. "
    "At the end, provide a clear summary recommendation."
)

# Memory for session dialogue
memory = ConversationBufferMemory(return_messages=True)
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
    system_message_template,
    ("human", "Based on the following context:\n{context}\n\nAnswer the user's question:\n{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)
retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

# --- Market Insight Engine: Fetch Live News ---
def get_market_news(keywords=["jobs", "career", "technology"]):
    """
    Fetch live news articles from NewsAPI (https://newsapi.org) and summarize.
    """
    url = (
        f"https://newsapi.org/v2/everything?q={' OR '.join(keywords)}"
        f"&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    )
    resp = requests.get(url)
    articles = resp.json().get("articles", [])
    texts = [f"{a['title']}: {a['description']}" for a in articles if 'title' in a and 'description' in a]
    if not texts:
        return "No recent market news found."
    # Use LLM to summarize articles into concise insights
    summary_prompt = (
        "Summarize the following news for recent market insight.\n" + "\n".join(texts)
    )
    news_summary = llm.invoke([{"role": "user", "content": summary_prompt}])
    return news_summary.content if hasattr(news_summary, "content") else str(news_summary)

# -- Personalized Weekly Newsletter Bot --
def send_newsletter(email, user_interests=["AI", "tech", "jobs"]):
    """
    Generate and send a newsletter with personalized market summaries.
    """
    news = get_market_news(user_interests)
    subject = "Weekly CareerBuilder AI Newsletter"
    body = f"Hello,\n\nHere are this week's market updates:\n\n{news}\n\nRegards,\nCareerBuilder AI"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = email
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, email, msg.as_string())
        server.quit()
        return "Newsletter sent!"
    except Exception as e:
        return f"Error sending email: {str(e)}"

# --- Flask Web UI ---
app = Flask(__name__)
app.secret_key = "super_secret_key"

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["chat_history"] = []
    memory.clear()  # If your ConversationBufferMemory has this method
    return '', 204  # No Content response (for Ajax), or redirect as you wish

@app.route("/", methods=["GET", "POST"])
def chat():
    session.setdefault("chat_history", [])
    newsletter_status = ""
    market_news = get_market_news()  # Refreshes with latest news

    # Handle user newsletter signup
    if request.method == "POST" and "user_email" in request.form:
        email = request.form["user_email"]
        user_interests = ["AI", "job", "technology"]
        newsletter_status = send_newsletter(email, user_interests)

    # Chat interaction
    if request.method == "POST" and "user_input" in request.form:
        user_msg = request.form["user_input"]
        session["chat_history"].append({"role": "User", "text": user_msg})

        # keywords = ["career", "job", "role", "skill"]
        # Always use retrieval chain—no more keyword branching!
        response = retrieval_chain.invoke(
            {
                "input": user_msg,
                "chat_history": memory.load_memory_variables({}).get("history", []),
            }
        )
        llm_response = response.get('answer', '')

        # if any(k in user_msg.lower() for k in keywords):
        #     response = retrieval_chain.invoke(
        #         {
        #             "input": user_msg,
        #             "chat_history": memory.load_memory_variables({}).get("history", []),
        #         }
        #     )
        #     llm_response = response.get('answer', '')
        # else:
        #     conversation_prompt = ChatPromptTemplate.from_messages(
        #         [
        #             system_message_template,
        #             MessagesPlaceholder(variable_name="history"),
        #             ("human", "{input}"),
        #         ]
        #     )
        #     chain = ConversationChain(llm=llm, memory=memory, prompt=conversation_prompt, verbose=True)
        #     llm_response = chain.invoke({"input": user_msg})['response']
        #     if not any(k in user_msg.lower() for k in keywords):
        #         llm_response += " Let's return to talking about your career interests!"

        memory.save_context({"input": user_msg}, {"output": llm_response})
        session["chat_history"].append({"role": "AI", "text": llm_response})

    return render_template(
        "index.html",
        chat_history=session["chat_history"],
        newsletter_status=newsletter_status,
        market_news=market_news,
    )

if __name__ == "__main__":
    app.run(debug=True)
