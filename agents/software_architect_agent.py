# In agents/software_architect_agent.py
from .base_agent import BaseAgent
from .utils import parse_json_from_response

class SoftwareArchitectAgent(BaseAgent):
    def __init__(self, llm_class, llm_args):
        super().__init__(llm_class, llm_args, temperature=0.3, agent_name="SoftwareArchitectAgent")

    def design_architecture(self, clarified_prompt, logger):
        system_message = """You are an expert, polyglot Software Architect. Your job is to design the high-level technical BLUEPRINT for a project. You define the structure and technology, you DO NOT define the build steps.

**Your Process:**
1.  Select the Technology Stack.
2.  Design the primary software components/modules (e.g., "Main Application", "Database Models", "User Interface Templates").
3.  For each component, list the files that will belong to it.
4.  Define the dependencies and run command.

**OUTPUT FORMAT:**
...
-   `"file_structure"`: A list of files.
-   `"dependencies"`: A dictionary of dependencies.
-   `"component_breakdown"`: A dictionary where keys are filenames and values are their purpose.
-   `"run_command"`: The command to run the app.

**Example for a Python Web App:**
{
  "technology_stack": "Python backend with Flask and SQLAlchemy, and a simple HTML/CSS frontend.",
  "file_structure": ["app.py", "models.py", "requirements.txt", "templates/index.html"],
  "dependencies": {
    "pip": ["Flask", "Flask-SQLAlchemy", "Flask-WTF"]
  },
  "component_breakdown": {
    "app.py": "Main Flask application file.",
    "models.py": "SQLAlchemy database models."
  },
  "run_command": "python app.py"
}

**Example for a TypeScript/Node.js Project:**
{
  "technology_stack": "TypeScript backend with Node.js and Express.",
  "file_structure": ["src/index.ts", "package.json", "tsconfig.json"],
  "dependencies": {
    "npm": ["express", "typescript", "ts-node", "@types/express", "@types/node"]
  },
  "component_breakdown": {
    "src/index.ts": "Main application entry point with Express server setup."
  },
  "run_command": "npx ts-node src/index.ts"
}
"""
        prompt = f"Here are the validated functional requirements for the project. Please design the technical architecture.\n\n{clarified_prompt}"

        response_str = self._call_llm(system_message, prompt, logger)
        
        data = parse_json_from_response(response_str)

        # --- THE FIX ---
        # The keys here must match the keys requested in the prompt's OUTPUT FORMAT section.
        required_keys = [
            "technology_stack",
            "file_structure",
            "dependencies",
            "component_breakdown",
            "run_command"
        ]

        if data and all(key in data for key in required_keys):
            # All required keys are present, the architecture is valid.
            return data
        else:
            print("❌ Error: SoftwareArchitectAgent failed to produce a valid architecture JSON.")
            if data:
                print(f"   Missing keys: {[key for key in required_keys if key not in data]}")
            else:
                print("   The response could not be parsed as JSON at all.")
            return None
