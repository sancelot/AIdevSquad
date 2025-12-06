
from .base_agent import BaseAgent
from .utils import parse_json_from_response
import json


class ProductOwnerAgent(BaseAgent):
    def __init__(self, llm_class, llm_args):
        super().__init__(llm_class, llm_args, temperature=0.2, agent_name="ProductOwnerAgent")

    async def chat(self, message: str) -> str:
        """
        Handle a chat message from another agent (e.g. DeveloperAgent).
        The PO acts as the guardian of the requirements and vision.
        """
        chat_system_message = """You are the Product Owner for this project.
        Your role is to clarify requirements, explain the vision, and make functional decisions when asked.
        
        You do NOT write code or create technical plans.
        You ARE the authority on WHAT needs to be built, not HOW.
        
        Answer the teammate's question specifically based on the project requirements.
        If the question is about technical implementation, suggest they ask the Architect or decide what's best for the user.
        """

        # We use the base chat capability but with a specific persona
        return self._call_llm(chat_system_message, message, None)
