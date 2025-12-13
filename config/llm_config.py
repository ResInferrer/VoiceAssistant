from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_llm(task: str) -> ChatOllama:
    """
    Returns a Large Language Model with certain parameters for the selected task. 
    
    :param task: The task that the model perform. A more suitable model is determined for a different task.
    :type task: str
    """
    if task == "general":
        return ChatOllama(
            model='qwen2.5:7b-instruct-q4_0',
            temperature=1,
            top_k=20,
            top_p=0.8,
            num_predict=512,
            num_thread=4,
            num_ctx=2048
            )
    elif task == "dispatcher":
        return ChatOllama(
            model='qwen2.5:7b-instruct-q4_0',
            temperature=0.2,
            top_k=20,
            top_p=0.8,
            num_predict=512,
            num_thread=4,
            num_ctx=512
            )
    else:
        print("exception") # Todo: Exception  

def get_prompt_template(task: str) -> ChatPromptTemplate:
    """
    Returns a promt template. In includes: system prompt, history, user input. 
    
    :param task: The task that the model perform. A more suitable model is determined for a different task.
    :type task: str
    """
    if task == "general":
        messages = [
            ("system", 
            """
                You are a helpful AI assistant. Provide accurate and honest information.
            """),
            MessagesPlaceholder("history"),
            ("human", "{user_input}")
        ]
        return ChatPromptTemplate.from_messages(messages) 
    elif task == "dispatcher":
        messages = [
            ("system", 
            """
                Task of dispatcher...
            """),
            MessagesPlaceholder("history"),
            ("human", "{user_input}")
        ]
        return ChatPromptTemplate.from_messages(messages) 
    else:
        print("exception") # Todo: Exception  