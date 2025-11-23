
from .base_agent import BaseAgent
from .utils import parse_json_from_response


class RequirementsAnalystAgent(BaseAgent):
    def __init__(self, llm_class, llm_args):
        super().__init__(llm_class, llm_args, temperature=0.2,
                         agent_name="RequirementsAnalystAgent")

    def analyze_requirements(self, user_prompt, conversation_history=""):
        """
        Analyzes the user prompt, identifies ambiguities, and asks clarifying questions.
        """
        system_message = """You are an expert Business Analyst and Requirements Analyst. Your sole purpose is to take a user's initial request and transform it into a clear, complete, and unambiguous set of functional requirements for a software project.

**YOUR PROCESS:**
1.  **Analyze the User's Goal:** Read the user's request and conversation history to understand the core business problem they are trying to solve.
2.  **Identify Functional Gaps:** Look for missing details about features, user interactions, or data. For example: "Does 'managing users' include password resets?", "What specific fields are needed for a 'contract'?", "Should there be a confirmation step before deleting an item?".
3.  **Propose Sensible Defaults:** For each functional gap, you MUST propose a standard, user-friendly default behavior. Your role is to guide the user towards a complete specification.
4.  **Formulate Clarifying Questions:** Frame your questions around your proposed defaults, asking the user for confirmation.
5.  **Refine the Requirements Document:** Create a "refined_prompt" that incorporates your proposed default choices. This document should describe WHAT the application does, not HOW it is built. Avoid technical terms like 'database', 'model', 'SQLite' unless the user has specifically mentioned them.

**OUTPUT FORMAT:**
Your response MUST be a single, valid JSON object with two keys:
-   `"questions"`: A list of strings. Each string is a question for the user, presenting a default choice for confirmation. If all requirements are clear, return an empty list `[]`.
-   `"refined_prompt"`: A single, well-structured string containing the detailed functional requirements.

**--- EXAMPLES ---**

**Example 1: Ambiguity Found**
*User Prompt:* "I need an app to track my contracts."

*Your JSON Output:*
{
  "questions": [
    "To properly track contracts, I suggest we include the following fields: 'client_name', 'start_date', 'end_date', and 'service_description'. Does this cover all the information you need?",
    "When a user deletes a contract, should the application ask for confirmation first (e.g., 'Are you sure?') to prevent accidental deletions? I recommend we add this for safety."
  ],
  "refined_prompt": "Build a web application to manage maintenance contracts. The application will allow users to store and view contracts. Each contract will have a 'client_name', 'start_date', 'end_date', and 'service_description'. It will feature a confirmation step before deleting a contract."
}

**Example 2: All Clear**
*User Prompt:* "I want a simple CRUD app for contracts with a client name and start date. Add a confirmation before deletion."

*Your JSON Output:*
{
  "questions": [],
  "refined_prompt": "Build a simple Create, Read, Update, Delete (CRUD) web application for managing contracts. Each contract must have a 'client_name' and a 'start_date'. The application must ask for user confirmation before permanently deleting a contract."
}
"""
        prompt = "Here is the user request and our conversation so far. Please analyze it.\n\n"
        if conversation_history:
            prompt += f"CONVERSATION HISTORY:\n{conversation_history}\n\n"

        prompt += f"CURRENT USER PROMPT:\n{user_prompt}"

        response_str = self._call_llm(system_message, prompt, logger=None)

        data = parse_json_from_response(response_str)
        if data and "questions" in data and "refined_prompt" in data:
            return data
        else:
            print("❌ Error: RequirementsAnalystAgent failed to produce a valid response.")
            # Return a "pass-through" response to avoid crashing
            return {"questions": [], "refined_prompt": user_prompt}
