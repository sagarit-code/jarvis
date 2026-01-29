import base64
from gmail_auth import get_gmail_service
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph
from dotenv import load_dotenv
import os
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
import json
from email.message import EmailMessage


load_dotenv()


service = get_gmail_service()

model=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


class GmailState(TypedDict):
    human_query:str
    main_intent:str
    writing_intent:dict
    email_of_person:str
    send_mail_draft:str
    send_mail_subject:str


def identifying_the_main_intent(state:GmailState) -> GmailState:
    prompt='''You are Jarvis, an intelligent Gmail assistant.Your task is to understand the user's primary intention.
    Classify the user's request into exactly ONE of the following categories:
    1. READING  
   - Reading emails
   - Summarizing emails
   - Searching emails
   - Labeling emails
   - Analyzing inbox
   - Finding important or urgent emails
   - Any task that DOES NOT send an email
   
   2. WRITING  
   - Drafting email replies
   - Sending emails
   - Writing follow-ups
   - Rewriting or improving an email
   - Any task that CREATES or SENDS an email
   
   Respond ONLY this text (reading or writing)!!!.'''
    
    messages=[
        SystemMessage(content=prompt),
        HumanMessage(content=state["human_query"])
    ]

    response=model.invoke(messages)
    answer=response.content

    state["main_intent"]=answer

    return state


def router(state:GmailState) -> GmailState:
    if state["main_intent"].lower()=="reading":
        return "reading"
    elif state["main_intent"].lower()=="writing":
        return "writing"
    else:
        return "end"
    
def intent_query_for_writing(state:GmailState) -> GmailState:
    prompt='''
    You are Jarvis, an AI Gmail writing intent classifier.
    
    Your task is to analyze the user's message and determine whether the user wants to:
    1) Reply to an existing email
    OR
    2) Write a new email

    IMPORTANT RULES:
    - This agent ONLY handles writing intents (sending, drafting, replying).
    - If the user wants to read, summarize, search, or analyze emails → return NOTHING.
    - You must return ONLY a valid JSON object.
    - Do NOT include explanations, markdown, or extra text.
    
    INTENT RULES:
    - If the user mentions replying, responding, answering, or says "reply to this email" → intent = "reply"
    - If the user wants to write or send an email to someone new → intent = "new_email"
    
    TO FIELD RULES:
    - Extract the recipient's name if mentioned (e.g., "email Rahul", "send to Aman")
    - If no person is clearly mentioned, set "to" = null
    - Do NOT guess names
    
    OUTPUT FORMAT (STRICT):
    {
    "to": "<name_or_null>",
    "intent": "reply" | "new_email"
    }'''

    messages=[
        SystemMessage(content=prompt),
        HumanMessage(content=state["human_query"])
        
    ]

    response=model.invoke(messages)
    answer=response.content

 
    state["writing_intent"] = json.loads(answer)

    return state

def preparing_draft(state:GmailState) -> GmailState:

    messages_to=service.users().messages().list(
        userId="me",
        q=f"from:{state['writing_intent']['to']}",
        maxResults=30
    ).execute()

    messages_id=[]
    for ids in messages_to["messages"]:
        messages_id.append(ids["id"])

    emails=set()
    for each in messages_id:
        details_id=service.users().messages().get(
            userId="me",
            id=each,
            format="full"
        ).execute()

        for details in details_id["payload"]["headers"]:
            if details["name"]=="From":
                if details["value"] not in emails:
                    emails.add(details["value"])

    if len(emails)==0:
        print("sagarit we cant find anyone in our inbox")
    elif len(emails)==1:
        email = list(emails)[0]
        b = email.split("<")
        ans=b[1]
        anss=ans[:-1]
        state["email_of_person"]=anss

    else:
        print("we have more than 1 emails with this name , can u specific the person full name")



    prompt_for_subject="you are an expert email writer, given the query generate the revelant and crisp email subject"
    messagess=[
        SystemMessage(content=prompt_for_subject),
        HumanMessage(content=state["human_query"])
    ]
    responsee=model.invoke(messagess)
    subject=responsee.content

    prompt_for_body="you are an expert email writer, given the query generate the revelant and crisp email body"
    messagess2=[
        SystemMessage(content=prompt_for_body),
        HumanMessage(content=state["human_query"])
    ]
    responsee2=model.invoke(messagess2)
    body=responsee2.content

    msg=EmailMessage()
    msg["From"]="me"
    msg["To"]=state["email_of_person"]
    msg["Subject"]=subject
    msg.set_content(body)

    state["send_mail_draft"]=body
    state["send_mail_subject"]=subject
    return state


def human_in_loop_send(state:GmailState) -> GmailState:
    print(f"here is the draft can we approve? {state['send_mail_draft']}")
    n = input("Enter your thoughts sir (type 'exit' to approve): ")
    while n != "exit":
        prompt = "here is what changes u need to do in this draft"
        mess = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"""
                draft:
                {state['send_mail_draft']}
                changes:
                {n}"""
            )
        ]

        response = model.invoke(mess)
        anss = response.content
        state["send_mail_draft"] = anss

        print(f"here is the draft can we approve? {state['send_mail_draft']}")
        n = input("Enter your thoughts sir (type 'exit' to approve): ")

    return state

def sending_final_draft(state:GmailState) -> GmailState:
    msg=EmailMessage()
    msg["From"]="me"
    msg["To"]=state["email_of_person"]
    msg["Subject"]=state["send_mail_subject"]
    msg.set_content(state["send_mail_draft"])

    encoded_msg_for_gmail=base64.urlsafe_b64encode(
        msg.as_bytes()
    ).decode()

    draft=service.users().drafts.create(
        userId="me",
        body={
            "raw":encoded_msg_for_gmail
        }
    ).execute()

    sending_draft=service.users().drafts.send(
        userId="me",
        body={"id":draft["id"]}
    ).execute()


graph=StateGraph(GmailState)
graph.add_node("main_intent",identifying_the_main_intent)
graph.add_node("writing_intent",intent_query_for_writing)

graph.add_edge(START,"main_intent")
graph.add_conditional_edges(
    "main_intent",
    router,
    {
        "reading":END,
        "writing":"writing_intent",
        "end":END
    }
)
graph.add_edge("writing_intent",END)


app=graph.compile()
query="send an email to rahul saying that schedule the meeting tomorrow"

result=app.invoke({"human_query":query})

print(result["main_intent"])
print(result["writing_intent"])

#workflows 

#-> taking the users query
#-> breaking the users query into intentions
   #a -- email reading (conditional nodes) 
   #b -- email labelling (conditional nodes)
   #c -- email sending/drafts/replies (conditional nodes)

   #note , a and b options are same comes under readding the emails
#-> each options has its known functions
#-> we will run each functions under the specific options
    #-> query filtering
    #-> reading/drafting/sending

