from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import HumanMessage, AIMessage
from graph.agent_workflow import create_agent_graph

def main():
    agent_graph = create_agent_graph()
    history = []

    while True:
        print()
        user_content = input('You: ')

        if user_content.lower() in ['q']:
            print("Выход...")
            break

        initial_state = {
            "messages": history.copy(),
            "user_input": user_content,
            "current_agent": "",
            "final_response": ""
        }
        
        full_response = ""
        print('AI: ', end='', flush=True)
        
        for chunk in agent_graph.stream(initial_state):
            for key, value in chunk.items():
                if hasattr(value, 'get') and value.get("final_response"):
                    response = value["final_response"]
                    if response != full_response:
                        new_text = response[len(full_response):]
                        if new_text:
                            print(new_text, end='', flush=True)
                            full_response = response
        
        print()
        
        history.append(HumanMessage(content=user_content))
        history.append(AIMessage(content=full_response))

if __name__ == "__main__":
    main()