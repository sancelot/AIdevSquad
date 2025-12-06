from .utils import call_llm


class BaseAgent:
    def __init__(self, llm_class, llm_args, temperature, agent_name):
        args = llm_args.copy()
        args['temperature'] = temperature
        self.llm = llm_class(**args)
        self.agent_name = agent_name

        model_display_name = llm_args.get(
            "model_name") or llm_args.get("model")
        print(
            f"  -> {self.agent_name} initialized with {model_display_name} (temp={self.llm.temperature})")

    def _call_llm(self, system_message, prompt, logger):
        return call_llm(self.llm, system_message, prompt, logger, self.agent_name)

    async def chat(self, message: str) -> str:
        """
        Generic chat method for inter-agent communication.
        Default implementation uses the basic LLM call.
        Subclasses (like ReAct agents) should override this to use their specific logic/tools.
        """
        # Default system message for a generic chat
        system_message = f"You are {self.agent_name}. Answer the user's question directly."
        # We use the synchronous _call_llm here.
        return self._call_llm(system_message, message, None)
