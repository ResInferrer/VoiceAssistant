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
                Ты ассистент-помощник. Твоя обязанность - помогать пользователю.
            """),
            MessagesPlaceholder("history"),
            ("human", "{user_input}")
        ]
        return ChatPromptTemplate.from_messages(messages) 

    elif task == "dispatcher":
        messages = [
            ("system", 
            """
                Роль: ты - диспетчер-маршрутизатор. Твоя единственная задача - проанализировать запрос пользователя и определить, какому агенту его передать.
                Правила:
                1. Анализ: Сравни смысл запроса пользователя со списком доступных задач агентов и их описания (ниже).
                2. Сопоставление: Если запрос точно соответствует описанию задачи одного из агентов - используй эту задачу.
                3. Выбор: Если подходит только одна задача - выбери её. Если подходят несколько задач - выбирай всех. Не повторяй одни и те же действия.
                4. Формат ответа - всегда строгий JSON. Никаких пояснений, только JSON-объект.

                Доступные задачи и агенты (в формате "название действия": "описание" -> "агент действия"):
                - test: "В запросе пользователя будет слово "тест" " -> test_agent

                Критически важные инструкции по формату:
                - "название действия" - брать строго из списка выше (например, input_user)!
                - "агент действия" - брать строго из списка выше (например, input_user_agent)!
                - Не придумывай новые названия и имена агентов.
                - Если запрос НЕ подходит ни под одну задачу - верни пустой объект: {{}}.

                Шаблон вывода (твой ответ - только это):
                {{"название действия": {{"agent": "агент действия"}}}}
            """),
            ("human", "{user_input}")
        ]
        return ChatPromptTemplate.from_messages(messages) 

    else:
        print("exception") # Todo: Exception  