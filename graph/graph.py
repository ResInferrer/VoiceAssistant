import json 

import pyaudio
from vosk import Model, KaldiRecognizer
from langgraph.graph import StateGraph, END
from config.llm_config import get_llm, get_prompt_template
from pyttsx3 import init

from state import AgentState


_WORKFLOW_COMPILE = False # todo: docs
_FIRST_RUNNING = True # todo: docs

# todo: docs
_AGENT_REGISTRY = {
    "test": 
        {
            "agent": "test_agent"
        }
}


class Graph:

    def create_agent_graph(self):
        workflow = StateGraph(AgentState) 
        workflow.add_node("orchestrator_agent", self._orchestrator_agent)
        workflow.add_node("plan_executor_agent", self._plan_executor_agent)
        workflow.add_node("user_input_agent", self._user_input_agent) 
        workflow.add_node("general_agent", self._general_agent) 
        workflow.add_node("tts_agent", self._tts_agent) 
        workflow.add_node("test_agent", self._test_agent)

        workflow.set_entry_point("user_input_agent")
        workflow.add_edge("user_input_agent", "orchestrator_agent")
        workflow.add_edge("orchestrator_agent", "plan_executor_agent")

        workflow.add_conditional_edges(
            "plan_executor_agent",
            self._route_after_plan_executor,
            {
                "test_agent": "test_agent",
                # ... modular agents

                "orchestrator_agent": "orchestrator_agent", # In case of any error in the plan recreate the plan
                "general_agent": "general_agent"
            }
        )

        workflow.add_edge("test_agent", "plan_executor_agent")
        # ... modular agents

        workflow.add_edge("general_agent", "tts_agent")
        workflow.add_edge("tts_agent", END)

        graph = workflow.compile()

        # Visualization. 
        # To see, go to the website https://mermaid.live/ 
        # and copy the code from agent_workflow_architecture.mmd
        global _WORKFLOW_COMPILE
        if not _WORKFLOW_COMPILE:
            mermaid_code = graph.get_graph().draw_mermaid()
            with open("agent_workflow_architecture.mmd", "w", encoding="utf-8") as f:
                f.write(mermaid_code)
            _WORKFLOW_COMPILE = True
        
        return graph

    def _user_input_agent(self, state: AgentState):
        global _FIRST_RUNNING
        if _FIRST_RUNNING:
            input_mode = input("Режим ввода text или voice: ")
        else:
            input_mode = state.get("input_mode")

        if input_mode == "voice":
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
        
            for user_text in listen():
                print(f"stt I: {user_text}")
            
                if user_text.lower() == "выход":
                    print("Выход...")
                    quit()
                else:
                    return {"user_input": user_text,
                            "input_mode": "voice"}
        else:
            user_input = input("text I: ")
            if user_input == "выход":
                print("Выход...")
                quit()
            else:
                return {"user_input": user_input,
                        "input_mode": "text"}


    def _orchestrator_agent(self, state: AgentState):
        """The Planner ..."""
        llm = get_llm("dispatcher")
        prompt_template = get_prompt_template(task="dispatcher")
        chain = prompt_template | llm

        user_input = state.get("user_input")
        response = chain.invoke({
        "user_input": user_input
        })
        
        print(f"response.content: {response.content}")

        return {"plan": json.loads(response.content)}

    def _plan_executor_agent(self, state: AgentState):
        """The Executor ..."""
        execution_plan = state.get("plan")
        if len(execution_plan) == 0:
            return {"current_agent": "general_agent"}

        keys = list(execution_plan.keys())
        current_step = keys[0]

        global _AGENT_REGISTRY
        keys_agent_registry = list(_AGENT_REGISTRY.keys())
        if current_step not in keys_agent_registry:
            print("error-plan_executor_agent") #todo: Exception

            return  {"current_agent": "orchestrator_agent"}

        agent_info = _AGENT_REGISTRY[current_step]
        current_agent = agent_info["agent"]

        first_key = next(iter(execution_plan))
        execution_plan.pop(first_key)

        return {"current_agent": current_agent}


    def _route_after_plan_executor(self, state: AgentState) -> str:
        current_agent = state.get("current_agent")
        return current_agent


    def _general_agent(self, state: AgentState):
        llm = get_llm(task="general")
        prompt_template = get_prompt_template(task="general")
        chain = prompt_template | llm

        response = chain.invoke({
            "history": state['messages'],
            "user_input": state.get("user_input")
        })
        
        return {"final_response": response.content}


    def _tts_agent(self, state: AgentState):
        final_text = state["final_response"]
        engine = init("sapi5") # for Windows

        # Set voice properties (optional)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 150) # two arg is speed of speech

        engine.say(final_text)
        engine.runAndWait()

        global _FIRST_RUNNING
        _FIRST_RUNNING = False

        return {"current_agent": "end"}


    def _test_agent(self, state: AgentState):
        print("test!")