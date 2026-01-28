import base64
from gmail_auth import get_gmail_service
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph
from dotenv import load_dotenv
import os
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

service = get_gmail_service()

model=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)





#nodes

#converting human response to gmail query parameters


