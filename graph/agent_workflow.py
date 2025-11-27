from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from config.llm_config import get_llm, get_prompt_template
from pyttsx3 import init
import operator

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # Conversation history
    user_input: str                                       # Current user data
    current_agent: str                                    # Which agent is currently working
    final_response: str
    mode: str

def orchestrator_agent(state: AgentState):
    # todo: llm logic
    mode = state.get("mode", "2")
    if mode == "1":
        return {"current_agent": "sst"}
    else:
        return {"current_agent": "general"}

def orchestrator_decision(state: AgentState):
    return state["current_agent"]

def sst_agent(state: AgentState):
    # sst logic
    test_input = input("(Голосовой режим) Введите текст для имитации: ")
    return {
        "current_agent": "general_agent",
        "user_input": test_input
    }

def general_agent(state: AgentState):
    llm = get_llm()
    prompt_template = get_prompt_template()
    chain = prompt_template | llm
    response = chain.invoke({
        "history": state['messages'],
        "user_input": state['user_input']
    })
    
    return {
        "final_response": response.content,
        "current_agent": "tts_agent"
    }


def tts_agent(state: AgentState):
    final_text = state["final_response"]
    engine = init("sapi5")

    # Set voice properties (optional)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 150) # Speed

    engine.say(final_text)
    engine.runAndWait()

    return {
        "current_agent": "end"
    }


def create_agent_graph():
    workflow = StateGraph(AgentState) 
    workflow.add_node("orchestrator_agent", orchestrator_agent) 
    workflow.add_node("sst_agent", sst_agent) 
    workflow.add_node("general_agent", general_agent) 
    workflow.add_node("tts_agent", tts_agent) 

    workflow.set_entry_point("orchestrator_agent")

    workflow.add_conditional_edges(
        "orchestrator_agent",
        orchestrator_decision,
        {
            "sst": "sst_agent",
            "general": "general_agent", 
            "end": END
        }
    )

    workflow.add_edge("sst_agent", "general_agent")
    workflow.add_edge("general_agent", "tts_agent")
    workflow.add_edge("tts_agent", END)

    # Visualization
    graph = workflow.compile()
    with open("agent_workflow_architecture.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    return graph