from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MODEL_NAME = 'qwen2.5:7b-instruct'
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=1
)

messages = [
    ("system", 
    """
        You are a helpful AI assistant. Provide accurate and honest information.
        Answer directly as an AI assistant.
    """),
    MessagesPlaceholder("history"),
]
prompt_template = ChatPromptTemplate(messages)

history = []
while True:
    print()
    user_content = input('You: ')

    if user_content.lower() in ['q']:
        print("Выход...")
        break

    history.append(HumanMessage(content=user_content))
    full_ai_content = ""
    print('Ai: ', end="")

    trimmed_history = trim_messages( 
        messages=history, 
        strategy="last",
        token_counter=len,
        max_tokens=500,
        include_system=False,
        allow_partial=False
    )
    
    chain = prompt_template | llm
    prompt_value = chain.invoke({"history": trimmed_history})
    for ai_message_chunk in llm.stream(prompt_value.to_messages()):
        print(ai_message_chunk.content, end="")
        full_ai_content += ai_message_chunk.content

    history.append(AIMessage(content=full_ai_content))
    print()