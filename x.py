from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START,END,StateGraph
from dotenv import load_dotenv
import os
import requests
from typing import TypedDict
import json

load_dotenv()

x_token=os.getenv("X_BEARER_TOKEN")
headers = {
    "Authorization": f"Bearer {x_token}",
    "Content-Type": "application/json"
}

model=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

class Xagent(TypedDict):
    human_q:str
    intent:dict
    content:dict

def intent_classifier(state:Xagent) -> Xagent:
    prompt='''

    You are Jarvis’s X (Twitter) Intelligence Agent.


    Your job is to understand a human’s natural language query about X (Twitter)
    and convert it into a STRICT JSON instruction set that another system
    will use to call the X API and analyze results.
    
    Rules:
    - Do NOT explain anything
    - Do NOT add extra text
    - Output ONLY valid JSON
    - Be concise but precise
    - Infer missing intent intelligently
    
    You must identify:
    1. What the user is trying to discover on X
    2. Keywords or topics to search
    3. Whether the user wants VIRAL content, GENERAL trends, or NICHE insights
    4. Time relevance (recent / last few days / timeless)
    5. Expected output format (topics, summaries, ideas, insights)

    If the user query is vague, assume they want:
    - Recent
    - High engagement
    - Relevant to their domain
    
    🧾 JSON SCHEMA (VERY IMPORTANT)
    
    must always output exactly this structure:

{
  "intent": "",
  "search_keywords": [],
  "topic_domain": "",
  "virality_filter": "",
  "time_range": "",
  "result_count": 5,
  "output_style": "",
  "summary_required": true/false
}

examples:

Human:

Jarvis, what’s going on X? What’s the viral topic related to AI agents?

LLM OUTPUT:
{
  "intent": "discover viral discussions on X",
  "search_keywords": ["AI agents", "autonomous agents", "AI workflows", "LLM agents"],
  "topic_domain": "AI",
  "virality_filter": "high_engagement",
  "time_range": "last_3_days",
  "result_count": 5,
  "output_style": "bullet_topics_with_insight",
  "summary_required": true
}

🧪 EXAMPLE 2 (Founder / Builder angle)
Human:

What AI agent ideas are founders talking about on X?

{
  "intent": "identify emerging ideas discussed by founders",
  "search_keywords": ["AI agent startup", "agent ideas", "AI automation", "founder builds"],
  "topic_domain": "startups",
  "virality_filter": "medium_to_high_engagement",
  "time_range": "last_7_days",
  "result_count": 7,
  "output_style": "idea_summary_with_examples",
  "summary_required": true
}

🧪 EXAMPLE 3 (Content inspiration)
Human:

Give me topics I can tweet about in AI that are doing well right now
{
  "intent": "content inspiration for posting on X",
  "search_keywords": ["AI agents", "LLMs", "AI productivity", "AI tools"],
  "topic_domain": "AI",
  "virality_filter": "high_engagement",
  "time_range": "last_48_hours",
  "result_count": 6,
  "output_style": "tweet_ready_topics",
  "summary_required": false
  }''' 
    messages=[
        SystemMessage(content=prompt),
        HumanMessage(content=state["human_q"])

    ]
    
    
    response=model.invoke(messages)
    ans=response.content

    state["intent"]=state["intent"] = json.loads(ans)
    return state





def finding_viral_posts(state:Xagent) -> Xagent:

    url="https://api.x.com/2/tweets/search/recent"

    params={
        "q":f"{state["intent"]["search_keywords"][0]}",
        "max_results":15,
        "tweet.fields":"public_metrics,author_id",
        "expansion":"author_id",
        "user.fields":"username,name,verified"
    }

    response=requests.get(url=url,params=params,headers=headers)
    ans=response.json()

    state["content"]=ans
    return state

graph=StateGraph(Xagent)
graph.add_node("intent",intent_classifier)
graph.add_node("discovery",finding_viral_posts)

graph.add_edge(START,"intent")
graph.add_edge("intent","discovery")
graph.add_edge("discovery",END)

app=graph.compile()

query="jarvis whats trending on x related to ai startups"

answer=app.invoke({"human_q":query})
print(answer["content"])




