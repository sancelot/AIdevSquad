
from .utils import parse_json_from_response
from .base_agent import BaseAgent
class DocumentationAgent(BaseAgent):

    def __init__(self, llm_class, llm_args):
        super().__init__(llm_class, llm_args, temperature=0.2, agent_name="DocumentationAgent")

    def write_documentation(self, project_name, project_structure, run_command, requirements_content,logger):
        """
        Generates a README.md file for the project.
        """
        system_message = f"""You are a senior technical writer. Your task is to create a clear and comprehensive README.md file for a new software project.

You will be given the project name, the file structure, the command to run the application, and the contents of the requirements file.

Structure your README.md as follows:
1.  **Project Title:** Use the provided project name.
2.  **Overview:** Write a brief, one-paragraph summary of what the application does based on the file structure and dependencies.
3.  **File Structure:** Present the provided file structure in a clean, readable format.
4.  **Prerequisites:** List the dependencies from the requirements file.
5.  **Installation & Setup:** Provide a step-by-step guide to set up the project:
    -   Create a virtual environment.
    -   Install dependencies using `pip install -r requirements.txt`.
    -   Mention if any database initialization is needed (e.g., if you see `db.create_all()` in the code, mention that the app creates it on first run).
6.  **Running the Application:** Clearly state the command needed to run the project.
7.  **Usage:** Briefly explain how to use the application (e.g., "Open your browser to http://127.0.0.1:5000...").

Generate only the Markdown content for the README.md file. Do not include any other text or explanations outside of the Markdown content.
"""

        prompt = f"""
Project Name: {project_name}

File Structure:
{project_structure}

Run Command:
`{run_command}`

Requirements (`requirements.txt`):
{requirements_content}
code
Code
"""
        
        # We can use the universal call_llm function
        readme_content = self._call_llm(system_message, prompt,logger)
        return readme_content
    