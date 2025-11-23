from .utils import call_llm 
class BaseAgent:
    def __init__(self, llm_class, llm_args, temperature, agent_name):
        args = llm_args.copy()
        args['temperature'] = temperature
        self.llm = llm_class(**args)
        self.agent_name = agent_name
        
        model_display_name = llm_args.get("model_name") or llm_args.get("model")
        print(f"  -> {self.agent_name} initialized with {model_display_name} (temp={self.llm.temperature})")
    
    def _call_llm(self, system_message, prompt, logger):
        return call_llm(self.llm, system_message, prompt, logger, self.agent_name)