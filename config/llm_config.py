from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_llm():
    return ChatOllama(
        model='qwen2.5:7b-instruct',
        temperature=1
    )

def get_prompt_template():
    messages = [
        ("system", 
        """
            You are a helpful AI assistant. Provide accurate and honest information.
            Answer directly as an AI assistant.
        """),
        MessagesPlaceholder("history"),
        ("human", "{user_input}")
    ]
    return ChatPromptTemplate.from_messages(messages) 