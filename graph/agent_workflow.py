from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from config.llm_config import get_llm, get_prompt_template

import operator

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # История разговоров
    user_input: str                                       # Текущие данные пользователей
    current_agent: str                                    # Какой агент сейчас работает
    final_response: str

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
        "current_agent": "end"
    }

def create_agent_graph():
    workflow = StateGraph(AgentState) 
    workflow.add_node("general_agent", general_agent) 
    workflow.set_entry_point("general_agent")
    workflow.add_edge("general_agent", END)

    # визуализация
    graph = workflow.compile()
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    return graph