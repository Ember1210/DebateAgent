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
DEFAULT_JUDGE2 = """
You are an expert judge of debates. Review the topic and the AFF case. Rate the strength of the AFF case as a number between 1 and 10 and return it as a parameter 'score'.
Also return the justification of your score, mentioning the value of the score you gave, as a parameter called 'response'. If score is 8 or higher, then the response should also declare Aff's victory.
Return the parameters combined as a JSON object.

"""

DEFAULT_AFF="You are an expert debater. Create a strong AFF case for the topic below."
DEFAULT_NEG="You are an expert debater. Create a strong NEG case for the topic below."

if "Judge" not in st.session_state:
   st.session_state['Judge']=DEFAULT_JUDGE

if "Judge2" not in st.session_state:
   st.session_state['Judge2']=DEFAULT_JUDGE2

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

class ScoreResponse(BaseModel):
   score: int
   response: str

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
  judge_pr2: str

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
    if (current_step == "Judgement" or current_step == "AffJudge"):
      return 'Judge'
    return END

  def judge(self, state: AgentState):
    print(f"Starting Judge with {state=}")
    current_step = state['step']
    if (current_step == "topic"):
      topic = state['topic']
      next_step = "AffOpen"
      return {"output": "Judge selected topic: " + topic, "debateTopic": topic, "step": next_step}
    elif current_step == "AffJudge":
      topic = state["debateTopic"]
      affCase = state["affCase"]
      pr = state["judge_pr2"]
      messageToLM = createLLMMessage(pr, [f"topic: {topic}", f"Aff Case: {affCase}"])
      responseFromLM = self.model.with_structured_output(ScoreResponse).invoke(messageToLM)
      if (responseFromLM.score >= 8):
         next_step = "Complete"
         return {'output': responseFromLM.response, 'step': next_step, 'judging': responseFromLM.response}
      else:
         next_step = "NegOpen"
         return {'output': responseFromLM.response, 'step': next_step}
      
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
      next_step = "AffJudge"
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
   judge_pr2 = st.text_area("Judge2",value=st.session_state['Judge2'])
   aff_pr=st.text_area("Aff",value=st.session_state['Aff'])
   neg_pr=st.text_area("Neg",value=st.session_state['Neg'])
   st.session_state['Judge']=judge_pr
   st.session_state['Judge2']=judge_pr2
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
  
  for s in app.graph.stream({'step': "topic","topic":topic,"judge_pr":judge_pr, "judge_pr2":judge_pr2, "aff_pr":aff_pr, "neg_pr":neg_pr}, thread):
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