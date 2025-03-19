# ✅ Install dependencies before running:
# !pip install -q fastapi uvicorn langchain langgraph langsmith groq requests transformers torch python-dotenv

# ✅ Import required libraries
import os
import getpass
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from transformers import pipeline

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Set API keys (Prompt user if not found in .env)
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter LangSmith API Key: ")

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter Groq API Key: ")

# ✅ Enable LangSmith tracing for debugging
os.environ["LANGSMITH_TRACING"] = "true"

# ✅ Initialize FastAPI app
app = FastAPI(title="Mental Health Chatbot", version="1.0")

# ✅ Initialize Groq-based Chat Model (Llama3-8B)
model = init_chat_model("llama3-8b-8192", model_provider="groq")

# ✅ Initialize Memory for LangGraph (Stores past chat history)
memory = MemorySaver()

# ✅ Define Chat Prompt Template (AI Personality)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a compassionate mental health chatbot. Respond with empathy."),
    MessagesPlaceholder(variable_name="messages"),
])

# ✅ Define LangGraph Function to Call the Model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

# ✅ Load Hugging Face Emotion Detection Model
sentiment_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# ✅ Function for Emotion Analysis
def analyze_emotions(text):
    """
    Analyzes emotions in the given text using a Hugging Face model.
    Returns a dictionary of emotions with confidence scores.
    """
    predictions = sentiment_model(text)  # Get multiple emotion predictions
    
    # Convert list of dictionaries into structured output
    emotions_dict = {emotion["label"]: round(emotion["score"], 3) for emotion in predictions[0]}

    # Get top emotion
    top_emotion = max(emotions_dict, key=emotions_dict.get)

    return emotions_dict, top_emotion

# ✅ Build LangGraph Workflow
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
app_chatbot = workflow.compile(checkpointer=memory)

# ✅ FastAPI Endpoint for Chatbot
@app.post("/chat/")
async def chat_with_bot(query: str):
    """
    Handles user queries:
    - Detects emotions in input text
    - Generates chatbot response using LangGraph
    """
    # Step 1: Analyze Emotions
    detected_emotions, top_emotion = analyze_emotions(query)

    # Step 2: Get Chatbot Response
    config = {"configurable": {"thread_id": "session_001"}}
    input_messages = [HumanMessage(query)]
    output = app_chatbot.invoke({"messages": input_messages}, config)

    return {
        "user_query": query,
        "detected_emotions": detected_emotions,
        "top_emotion": top_emotion,
        "chatbot_response": output["messages"][-1].content
    }

# ✅ Run the FastAPI app using:
# uvicorn main:app --host 0.0.0.0 --port 8000
