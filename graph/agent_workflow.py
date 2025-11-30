import operator
import json 

import pyaudio
from vosk import Model, KaldiRecognizer
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from config.llm_config import get_llm, get_prompt_template
from pyttsx3 import init

#_VISUALIZATION_DONE = True

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # Conversation history
    user_input: str                                       # Current user data
    current_agent: str                                    # Which agent is currently working
    final_response: str                                   # Ready and final answer from LLM
    mode: str                                             # either voice input (2) or text input (1) from the user

def orchestrator_agent(state: AgentState):
    # todo: llm logic
    mode = state.get("mode", "1")
    if mode == "1":
        return {"current_agent": "text"}
    else:
        return {"current_agent": "sst"}

def orchestrator_decision(state: AgentState):
    return state["current_agent"]


def user_input_sst_agent(state: AgentState):
    model = Model('vosk_models/vosk-model-small-ru-0.22')
    rec = KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    def listen():
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if (rec.AcceptWaveform(data)) and (len(data)) > 0:
                answer = json.loads(rec.Result())
                if answer['text']:
                    yield answer['text']
    
    for text in listen():
        print(f"sst I: {text}")
        
        if text.lower() == "выход":
            print("Выход...")
            quit()
        else:
            return {"user_input": text}

def user_input_text_agent(state: AgentState):
    user_input = input("text I: ")
    if user_input == "выход":
        print("Выход...")
        quit()
    else:
        return {"user_input": user_input}


def general_agent(state: AgentState):
    llm = get_llm()
    prompt_template = get_prompt_template()
    chain = prompt_template | llm

    response = chain.invoke({
        "history": state['messages'],
        "user_input": state.get("user_input")
    })
    
    return {"final_response": response.content}


def tts_agent(state: AgentState):
    final_text = state["final_response"]
    engine = init("sapi5") # for Windows

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
    workflow.add_node("user_input_sst_agent", user_input_sst_agent) 
    workflow.add_node("user_input_text_agent", user_input_text_agent)
    workflow.add_node("general_agent", general_agent) 
    workflow.add_node("tts_agent", tts_agent) 

    workflow.set_entry_point("orchestrator_agent")

    workflow.add_conditional_edges(
        "orchestrator_agent",
        orchestrator_decision,
        {
            "sst": "user_input_sst_agent",
            "text": "user_input_text_agent", 
            "end": END
        }
    )

    workflow.add_edge("user_input_sst_agent", "general_agent")
    workflow.add_edge("user_input_text_agent", "general_agent")
    workflow.add_edge("general_agent", "tts_agent")
    workflow.add_edge("tts_agent", END)

    # Visualization
    #if _VISUALIZATION_DONE:
    #    with open("agent_workflow_architecture.png", "wb") as f:
    #        f.write(graph.get_graph().draw_mermaid_png())
    #    _VISUALIZATION = False
    
    graph = workflow.compile()
    return graph