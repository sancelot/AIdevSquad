
from .base_agent import BaseAgent
from .utils import parse_json_from_response
import json


class ProductOwnerAgent(BaseAgent):
    def __init__(self, llm_class, llm_args):
        super().__init__(llm_class, llm_args, temperature=0.2, agent_name="ProductOwnerAgent")

    def create_plan(self,  functional_prompt, technical_architecture, logger):
        system_message = """You are an expert Product Owner who creates development plans. You will be given a user's request and a technical architecture from the Software Architect.

Your job is to create a build plan, BUT FIRST, you must **validate the architecture against the user's request.**
-   Does the architecture's `dependencies` list include all the libraries needed for the features (e.g., forms need `Flask-WTF`, databases need `Flask-SQLAlchemy`)?
-   Is the `file_structure` consistent?

If the architecture is flawed, your FIRST task in the plan should be to fix it.

Example Plan with a fix:
1.  "FIX: The architecture is missing `Flask-SQLAlchemy` in `requirements.txt`. Add it."
2.  "Create `requirements.txt`..."


**Your Process:**
1.  Look at the architecture's `dependencies`. Your FIRST task must always be to create the dependency file (e.g., `requirements.txt`).
2.  Look at the `component_breakdown`. Determine the logical order to build these files.
3.  Create a task for each logical step. Break down large files into multiple tasks. For example, instead of one task for "Create app.py", create several:
    -   "Create `app.py` with basic Flask setup and database initialization."
    -   "Add the `Contract` model definition to `models.py`."
    -   "Implement the create contract route in `app.py`."
    -   "Implement the list contracts route in `app.py`."
4.  Your final plan should be a list of small, atomic tasks.

**OUTPUT FORMAT:**
Your response MUST be a JSON object with "plan" and "run_command" keys. The "plan" should be a list of small, atomic tasks.
"""
        prompt = f"""
**Functional Requirements:**
{functional_prompt}

**Technical Architecture to implement:**
```json
{json.dumps(technical_architecture, indent=2)}
```

"""

        response_str = self._call_llm(system_message, prompt, logger)

        data = parse_json_from_response(response_str)
        if data and "plan" in data and "run_command" in data:
            return data["plan"], data["run_command"]
        else:
            print(
                "Error: The PO did not return a valid JSON plan with the required keys.")
            return [], ""  # Return empty to signal failure

         # NOUVELLE MÉTHODE : REPLAN_PROJECT

    def replan_project(self, functional_prompt, original_plan, completed_tasks, failed_task, issues, logger):
        system_message = """You are an expert Agile Product Owner, responsible for maintaining and adapting the project plan.
A task has failed its code review due to complex issues, and the original plan is now invalid. Your job is to create a new, revised plan to address the failure and get the project back on track.

**YOUR PROCESS:**
1.  **Analyze the Failure:** Understand WHY the `failed_task` did not pass the review by looking at the `issues`. The issue description is the problem, the suggestion is a hint.
2.  **Review the Context:** Look at the `original_plan` and the `completed_tasks` to understand the project's state.
3.  **Create a Corrective Action Plan:** The core of your job is to break down the solution for the `issues` into a series of new, small, atomic tasks.
4.  **Integrate and Re-sequence:** Integrate these new corrective tasks into the original plan. You MUST place them at the correct position. This usually means right after the last completed task. Remove or modify any future tasks that are now redundant or incorrect due to the failure.
5.  **Produce a New, Complete Plan:** Your output must be a single, coherent, and complete list of all remaining tasks in the correct order.

**EXAMPLE:**
-   **Failed Task:** "Implement POST route"
-   **Issue:** "The route lacks actual contract creation logic"
-   **Your thought process:** The developer only created an empty shell. To fix this, we need several steps: 1. Import the model and db session. 2. Get data from the request. 3. Create a model instance. 4. Add to the session and commit.
-   **Your new plan might look like:**
    -   (Task that just completed)
    -   "FIX: Import `Contrats` model and `db` object into `app.py`."  <- NEW
    -   "FIX: In `create_contract` route, retrieve form data from the request." <- NEW
    -   "FIX: In `create_contract` route, create a `Contrats` object and save it to the database." <- NEW
    -   "Implement GET route to display all contracts..." (original next task)
    -   ... (rest of the original plan)

**OUTPUT FORMAT:**
Your response MUST be a JSON object with a single key "new_plan", which is a list of strings representing the new full plan of REMAINING tasks.
"""
        # On passe toutes les tâches complétées pour le contexte
        completed_tasks_str = "\n- ".join(completed_tasks)
        # On passe le plan des tâches restantes
        remaining_plan = [t for t in original_plan if t not in completed_tasks]
        remaining_plan_str = "\n- ".join(remaining_plan)

        prompt = f"""
**Original Functional Requirements:**
{functional_prompt}

**Project State:**
-   **Tasks Already Completed Successfully:**
    - {completed_tasks_str}
-   **The Plan for Future Tasks Was:**
    - {remaining_plan_str}

**CRITICAL FAILURE:**
The following task has just failed its code review:
-   **Failed Task:** "{failed_task}"

**Reasons for Failure (from Code Reviewer):**
```json
{json.dumps(issues, indent=2)}
```

Based on this failure, revise the entire future plan. Decompose the solution into small, atomic tasks and integrate them correctly. Provide the new, complete plan of all remaining tasks.
"""

        response_str = self._call_llm(system_message, prompt, logger)
        data = parse_json_from_response(response_str)
        if data and "new_plan" in data and isinstance(data["new_plan"], list):
            return data["new_plan"]
        else:
            self.logger.log(
                "ERROR", "ProductOwnerAgent failed to generate a valid new plan. Halting.")
            return None  # Signal failure
