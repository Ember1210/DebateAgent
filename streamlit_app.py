import streamlit as st
import os
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"]="DebateAgent"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
import random

st.set_page_config(layout = "wide")

DEFAULT_JUDGE="You are an expert Judge of debates. Review the topic, AFF and NEG case below and delivery your verdict, with explanation."
DEFAULT_AFF="You are an expert debater. Create a strong AFF case for the topic below."
DEFAULT_NEG="You are an expert debater. Create a strong NEG case for the topic below."

if "Judge" not in st.session_state:
   st.session_state['Judge']=DEFAULT_JUDGE

if "Aff" not in st.session_state:
   st.session_state['Aff']=DEFAULT_AFF

if "Neg" not in st.session_state:
   st.session_state['Neg']=DEFAULT_NEG

def createLLMMessage(prompt: str, messages: List):
  results = []
  results.append(SystemMessage(prompt))
  for msg in messages:
    results.append(HumanMessage(msg))
  return results

class AgentState(TypedDict):
  agent: str
  affCase: str
  negCase: str
  judging: str
  output: str
  step: str
  topic: str
  debateTopic: str
  aff_pr: str
  neg_pr: str
  judge_pr: str

class debateAgent:
  def __init__(self):
    self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key = st.secrets['OPENAI_API_KEY'])
    workflow = StateGraph(AgentState)
    workflow.add_node("Judge", self.judge)
    workflow.add_node("Aff", self.aff)
    workflow.add_node("Neg", self.neg)
    workflow.add_edge(START, "Judge")
    workflow.add_conditional_edges("Judge", self.router)
    workflow.add_conditional_edges("Aff", self.router)
    workflow.add_conditional_edges("Neg", self.router)
    self.graph = workflow.compile()

  def router(self, state: AgentState):
    print(f"inside router with {state=}")
    current_step = state['step']
    if (current_step == "AffOpen"):
      return 'Aff'
    if (current_step == "NegOpen"):
      return 'Neg'
    if (current_step == "Judgement"):
      return 'Judge'
    return END

  def judge(self, state: AgentState):
    print(f"Starting Judge with {state=}")
    current_step = state['step']
    if (current_step == "topic"):
      topic = state['topic']
      next_step = "AffOpen"
      return {"output": "Judge selected topic: " + topic, "debateTopic": topic, "step": next_step}
    else:
      next_step = END
      topic = state["debateTopic"]
      affCase = state["affCase"]
      negCase = state["negCase"]
      pr = state["judge_pr"]
      messageToLM = createLLMMessage(pr, [f"topic: {topic}", f"Aff Case: {affCase}", f"Neg Case: {negCase}"])
      responseFromLM = self.model.invoke(messageToLM)
      resp = responseFromLM.content

      return {"output": "Judge selected a winner", "step": next_step, "judging": resp}

  def aff(self, state: AgentState):
    print(f"Starting Aff with {state=}")
    current_step = state['step']
    topic = state["debateTopic"]
    pr = state["aff_pr"]
    if (current_step == "AffOpen"):
      argument = f"Affirmative Case for {topic}"
      next_step = "NegOpen"
      messageToLM = createLLMMessage(pr, [f"topic: {topic}"])
      responseFromLM = self.model.invoke(messageToLM)
      resp = responseFromLM.content
    return {"output": "I am aff", "step": next_step, "affCase": resp}

  def neg(self, state: AgentState):
    print(f"Starting Neg with {state=}")
    current_step = state['step']
    topic = state["debateTopic"]
    pr = state["neg_pr"]
    if (current_step == "NegOpen"):
      argument = f"Negative Case for {topic}"
      next_step = "Judgement"
      messageToLM = createLLMMessage(pr, [f"topic: {topic}"])
      responseFromLM = self.model.invoke(messageToLM)
      resp = responseFromLM.content
    return {"output": "I am neg", "step": next_step, "negCase": resp}

st_topic=st.empty()

with st.sidebar:
   judge_pr=st.text_area("Judge",value=st.session_state['Judge'])
   aff_pr=st.text_area("Aff",value=st.session_state['Aff'])
   neg_pr=st.text_area("Neg",value=st.session_state['Neg'])
   st.session_state['Judge']=judge_pr
   st.session_state['Aff']=aff_pr
   st.session_state['Neg']=neg_pr

col1,col2,col3=st.columns(3, border=True)
col1.header("Judge")
col1msg=col1.empty()
col2.header("Aff")
col3.header("Neg")


topic = st_topic.text_input("debateTopic")
if topic:
  app = debateAgent()
  thread_id=random.randint(1000, 9999)
  thread={"configurable":{"thread_id":thread_id}}
  
  for s in app.graph.stream({'step': "topic","topic":topic,"judge_pr":judge_pr, "aff_pr":aff_pr, "neg_pr":neg_pr}, thread):
    print(f"DEBUG {s=}")
    #for k,v in s.items():
    #    if resp := v.get("output"):
    #        print(f"**** {k=}, {resp=}")
    for k,v in s.items():
        if k=='Judge':
            if v.get('step')=='AffOpen':
                col1msg.write(v.get('output'))
            else:
                col1msg.write(v.get("judging"))
        if k=='Aff':
            col2.write(v.get("affCase"))
        if k=='Neg':
            col3.write(v.get("negCase"))
  print("COMPLETED")