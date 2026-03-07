from typing import TypedDict, Annotated, Dict
import operator

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):

    messages: Annotated[list[BaseMessage], operator.add]  # Conversation history
    user_input: str                                       # Current user data
    current_agent: str                                    # Which agent is currently working
    final_response: str                                   # Ready and final answer from LLM
    plan: Dict[str, Dict[str, str]]                       # The plan from LLM. It is necessary for a structured execution plan in 
    input_mode: str