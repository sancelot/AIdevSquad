
from .base_agent import BaseAgent
from .utils import parse_json_from_response
import json


class SADTSARTPlannerAgent(BaseAgent):
    def __init__(self, llm_class, llm_args):
        super().__init__(llm_class, llm_args, temperature=0.2,
                         agent_name="SADTSARTPlannerAgent")

    def generate_workplan(self, project_title, project_context, logger=None):
        system_message = """You are an expert systems engineer. Transform the following PROJECT into a hierarchical workplan using LLM-SADT/SART.
Produce a JSON object with this structure:
{
  "project_title": string,
  "context": string,
  "tasks": [
    {
      "id": "T1",
      "title": string,
      "description": string,
      "icom": {"input": [..], "control": [..], "output": [..], "mechanism": [..]},
      "subtasks": [ recursive tasks ]
    }
  ]
}
Rules:
- Keep tree depth <= 4 and split tasks until each atomic task is roughly a single LLM prompt-sized action (~50-250 tokens of work).
- Provide meaningful IDs (T1, T1.1, T1.1.1, ...)
- For each task fill icom fields concisely.
- Provide at most 8 top-level tasks.
- Output only valid JSON. Do not include any text outside the JSON block.
"""
        prompt = f"PROJECT TITLE: {project_title}\n\nPROJECT CONTEXT:\n{project_context}"

        max_retries = 3
        for attempt in range(max_retries):
            response_str = self._call_llm(system_message, prompt, logger)
            plan = parse_json_from_response(response_str)
            if plan:
                return plan

            if logger:
                logger.log(
                    "WARNING", f"Attempt {attempt+1} failed to parse JSON workplan. Retrying...")
            prompt += f"\n\nPREVIOUS ATTEMPT FAILED. Please ensure the output is strictly valid JSON. Do not add trailing commas or comments."

        if logger:
            logger.log(
                "ERROR", "Failed to generate valid JSON workplan after multiple attempts.")
        return None

    def generate_atomic_actions_for_task(self, task, logger=None):
        system_message = """You are given a TASK JSON object with fields (id, title, description, icom).
Return a JSON array of ATOMIC ACTIONS where each action is:
{
  "action_id": "T1.2.1.a",
  "task_id": "T1.2.1",
  "instruction": string (single instruction the LLM can execute),
  "constraints": [..],
  "expected_output_description": string
}
Rules:
- Keep each instruction short and deterministic.
- Prefer actions that can be executed by a single LLM call.
- Output only JSON array.
"""
        # We only pass the relevant fields to avoid token limit issues if the task object is huge
        task_subset = {
            "id": task.get("id"),
            "title": task.get("title"),
            "description": task.get("description"),
            "icom": task.get("icom")
        }
        prompt = f"TASK:\n{json.dumps(task_subset, ensure_ascii=False, indent=2)}"

        max_retries = 3
        for attempt in range(max_retries):
            response_str = self._call_llm(system_message, prompt, logger)
            actions = parse_json_from_response(response_str)
            if actions:
                return actions

            if logger:
                logger.log(
                    "WARNING", f"Attempt {attempt+1} failed to parse JSON atomic actions. Retrying...")
            prompt += f"\n\nPREVIOUS ATTEMPT FAILED. Please ensure the output is strictly valid JSON."

        return None

    def _flatten_tasks(self, tasks, parent_id=""):
        flat_tasks = []
        for t in tasks:
            flat_tasks.append(t)
            if t.get("subtasks"):
                flat_tasks.extend(self._flatten_tasks(
                    t["subtasks"], t.get("id", "")))
        return flat_tasks

    def create_plan(self, functional_prompt, technical_architecture, logger):
        """
        Orchestrates the SADT/SART planning process.
        Returns a list of atomic tasks (as strings) and a run command.
        """
        project_title = "Software Development Project"  # Could be extracted or passed

        # Combine functional prompt and architecture into the context
        project_context = f"""
Functional Requirements:
{functional_prompt}

Technical Architecture:
{json.dumps(technical_architecture, indent=2)}
"""

        logger.log("INFO", "Generating SADT/SART hierarchical workplan...")
        workplan = self.generate_workplan(
            project_title, project_context, logger)

        if not workplan:
            logger.log("ERROR", "generate_workplan returned None.")
            return [], ""

        if "tasks" not in workplan:
            logger.log(
                "ERROR", f"Workplan missing 'tasks' key. Keys found: {workplan.keys()}")
            return [], ""

        # Flatten the hierarchical tasks to process them linearly for atomic action generation
        all_tasks = self._flatten_tasks(workplan.get("tasks", []))

        logger.log("INFO", f"Flattened tasks count: {len(all_tasks)}")

        final_plan_strings = []

        logger.log(
            "INFO", f"Generated {len(all_tasks)} tasks in the hierarchy. Converting to atomic actions...")

        for task in all_tasks:
            # If a task has subtasks, it might be a container.
            if task.get("subtasks") and len(task.get("subtasks")) > 0:
                # logger.log("INFO", f"Skipping container task {task.get('id')}")
                continue

            # logger.log("INFO", f"Processing leaf task {task.get('id')}")
            actions = self.generate_atomic_actions_for_task(task, logger)

            if not actions:
                # Fallback if LLM fails
                logger.log(
                    "WARNING", f"No actions generated for {task.get('id')}, using fallback.")
                actions = [{
                    "action_id": f"{task.get('id')}.a",
                    "instruction": task.get('description') or task.get('title'),
                    "constraints": task.get('icom', {}).get('control', []),
                    "expected_output_description": task.get('icom', {}).get('output', ["result"])[0]
                }]

            if isinstance(actions, list):
                for action in actions:
                    # Format the atomic action into a string for the DeveloperAgent
                    # We include constraints and expected output to guide the developer
                    constraints_str = ", ".join(action.get("constraints", [])) if isinstance(
                        action.get("constraints"), list) else str(action.get("constraints"))

                    task_str = f"[{action.get('action_id')}] {action.get('instruction')}"
                    if constraints_str:
                        task_str += f" (Constraints: {constraints_str})"
                    if action.get("expected_output_description"):
                        task_str += f" (Output: {action.get('expected_output_description')})"

                    final_plan_strings.append(task_str)

        logger.log("INFO", f"Final plan has {len(final_plan_strings)} items.")

        # Determine a run command.
        run_command = "python app.py"  # Default fallback
        if "flask" in str(technical_architecture).lower():
            run_command = "python app.py"
        elif "node" in str(technical_architecture).lower() or "react" in str(technical_architecture).lower():
            run_command = "npm start"

        return final_plan_strings, run_command
