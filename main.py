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
        
        result = agent_graph.invoke(initial_state)
        
        ai_response = result["final_response"]
        print('Ai:', ai_response)
        
        history.append(HumanMessage(content=user_content))
        history.append(AIMessage(content=ai_response))

if __name__ == "__main__":
    main()