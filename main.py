from langchain_core.messages import HumanMessage, AIMessage

from graph.graph import Graph


def main():

    agent_graph = Graph.create_agent_graph()
    history = []

    while True:
        print()

        initial_state = {
            "messages": history.copy(),
            "user_input": "",
            "current_agent": "",
            "final_response": "",
            "plan": "",
            "input_mode": ""
        }
        
        full_response = ""
        
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
        
        #history.append(HumanMessage(content=user_content))
        history.append(AIMessage(content=full_response))

if __name__ == "__main__":
    main()