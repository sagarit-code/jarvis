# =========================
# ALL-IN-ONE JARVIS GMAIL AGENT
# =========================

import os
import base64
from typing import TypedDict, List

from dotenv import load_dotenv
from gmail_auth import get_gmail_service

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph

# =========================
# ENV + CLIENTS
# =========================

load_dotenv()

service = get_gmail_service()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# SYSTEM PROMPTS
# =========================

QUERY_SYSTEM_PROMPT = """You are a Gmail search query expert.

Your ONLY task is to convert a user’s natural language request into a valid
Gmail search query string that can be used directly as the `q` parameter
in the Gmail API (`users.messages.list`).

STRICT RULES:
- Output ONLY the Gmail query string
- No explanations
- No JSON
- No hallucinated filters

ALLOWED OPERATORS:
from:, to:, subject:, is:read, is:unread,
newer_than:Xd, older_than:Xd,
has:attachment, filename:, category:, label:
Use OR only if user explicitly says "or"

TIME WORDS:
today → newer_than:1d
yesterday → newer_than:2d
last week → newer_than:7d
last month → newer_than:30d

Numbers NEVER mean time.
"""

SUMMARY_SYSTEM_PROMPT = """You are an email summarization assistant.
Summarize the email in 2–3 concise bullet points.
Ignore signatures, disclaimers, and unsubscribe text.
"""

# =========================
# STATE
# =========================

class GmailState(TypedDict):
    human_query: str
    gmail_query: str
    message_ids: List[str]
    subjects: List[str]
    summaries: List[str]

# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------

def decode_base64(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode()).decode("utf-8", errors="ignore")


def extract_body_from_payload(payload: dict) -> List[dict]:
    """
    Recursively extracts all text/plain and text/html parts.
    Returns list of {mime, content}
    """
    results = []

    # CASE 1: payload.body exists and has data
    body = payload.get("body", {})
    if body.get("data"):
        results.append({
            "mime": payload.get("mimeType"),
            "content": decode_base64(body["data"])
        })

    # CASE 2: payload.parts exists → recurse
    for part in payload.get("parts", []):
        results.extend(extract_body_from_payload(part))

    return results


def get_best_email_text(payload: dict) -> str:
    bodies = extract_body_from_payload(payload)

    # Priority: plain text > html
    for b in bodies:
        if b["mime"] == "text/plain":
            return b["content"]

    for b in bodies:
        if b["mime"] == "text/html":
            return b["content"]

    return ""

# ------------------------------------------------------------------
# NODES
# ------------------------------------------------------------------

def generate_query(state: GmailState) -> GmailState:
    messages = [
        SystemMessage(content=QUERY_SYSTEM_PROMPT),
        HumanMessage(content=state["human_query"])
    ]
    response = model.invoke(messages)
    state["gmail_query"] = response.content.strip()
    print(response.content)
    return state


def fetch_message_ids(state: GmailState) -> GmailState:
    response = service.users().messages().list(
        userId="me",
        q=state["gmail_query"],
        maxResults=3
    ).execute()

    state["message_ids"] = [m["id"] for m in response.get("messages", [])]
    return state


def extract_and_summarize(state: GmailState) -> GmailState:
    subjects = []
    summaries = []

    for msg_id in state["message_ids"]:
        message = service.users().messages().get(
            userId="me",
            id=msg_id,
            format="full"
        ).execute()

        payload = message["payload"]

        # SUBJECT
        subject = "(No Subject)"
        for h in payload.get("headers", []):
            if h["name"] == "Subject":
                subject = h["value"]
                break

        # BODY
        email_text = get_best_email_text(payload)
        if not email_text.strip():
            continue

        # SUMMARY
        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=email_text[:8000])
        ]
        result = model.invoke(messages)

        subjects.append(subject)
        summaries.append(result.content.strip())

    state["subjects"] = subjects
    state["summaries"] = summaries
    return state

# ------------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------------

graph = StateGraph(GmailState)

graph.add_node("query", generate_query)
graph.add_node("fetch_ids", fetch_message_ids)
graph.add_node("extract", extract_and_summarize)

graph.add_edge(START, "query")
graph.add_edge("query", "fetch_ids")
graph.add_edge("fetch_ids", "extract")
graph.add_edge("extract", END)

app = graph.compile()

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------

if __name__ == "__main__":
    user_input = "hey bro summarize my 3 latest unread emails"

    result = app.invoke({
        "human_query": user_input
    })

   

    print("\nSUMMARIES:")
    for sm in result["summaries"]:
        print("-", sm)