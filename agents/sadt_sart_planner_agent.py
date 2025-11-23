
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
- Output only valid JSON.
"""
        prompt = f"PROJECT TITLE: {project_title}\n\nPROJECT CONTEXT:\n{project_context}"

        response_str = self._call_llm(system_message, prompt, logger)
        return parse_json_from_response(response_str)

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

        response_str = self._call_llm(system_message, prompt, logger)
        return parse_json_from_response(response_str)

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

        if not workplan or "tasks" not in workplan:
            logger.log("ERROR", "Failed to generate workplan.")
            return [], ""

        # Flatten the hierarchical tasks to process them linearly for atomic action generation
        # Note: In a real SART flow, we might respect the hierarchy more, but for the orchestrator's linear execution, we flatten.
        # However, we only want to generate atomic actions for the LEAF nodes or process all nodes?
        # The user's example flattens ALL tasks and generates actions for them.
        # But usually, only leaf tasks are executed.
        # Let's assume we generate actions for all tasks in the flattened list,
        # but the prompt implies "split tasks until each atomic task is roughly a single LLM prompt-sized action".
        # So the leaf nodes of the workplan are the ones we really care about for execution.
        # But the user's code does: `tasks = flatten_tasks_to_list(plan.get("tasks", []))` and then `generate_atomic_actions_for_task(t)` for EACH task.
        # This implies even high-level tasks might have atomic actions (maybe coordination actions?).
        # Let's follow the user's example: flatten all tasks and generate actions for each.

        all_tasks = self._flatten_tasks(workplan.get("tasks", []))

        final_plan_strings = []

        logger.log(
            "INFO", f"Generated {len(all_tasks)} tasks in the hierarchy. Converting to atomic actions...")

        for task in all_tasks:
            # If a task has subtasks, it might be a container.
            # If we generate actions for it, they might be redundant with subtasks.
            # But let's stick to the user's logic.
            # Optimization: If a task has subtasks, maybe we skip generating actions for IT directly,
            # and only generate for the subtasks?
            # The user's prompt says "split tasks until each atomic task is..."
            # If I have T1 and T1.1, T1.2. T1 is the parent.
            # If I execute T1 actions AND T1.1 actions, I might duplicate work.
            # Let's check if the task has subtasks. If yes, we might skip it and rely on subtasks.
            if task.get("subtasks") and len(task.get("subtasks")) > 0:
                continue

            actions = self.generate_atomic_actions_for_task(task, logger)

            if not actions:
                # Fallback if LLM fails
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

        # Determine a run command.
        # The PO agent usually guesses this. We can ask the LLM or default to something.
        # For now, let's try to extract it or just return a placeholder.
        # We can add a small step to ask for the run command if needed, or just infer it.
        # Let's infer it from the architecture or just leave it empty for the user to provide/dev agent to figure out.
        # Existing PO agent does: returns `data["run_command"]`.
        # Let's add a quick heuristic or LLM call for the run command if we want to be fully compatible.
        # Or just hardcode "python app.py" / "npm start" based on architecture.

        run_command = "python app.py"  # Default fallback
        if "flask" in str(technical_architecture).lower():
            run_command = "python app.py"
        elif "node" in str(technical_architecture).lower() or "react" in str(technical_architecture).lower():
            run_command = "npm start"

        return final_plan_strings, run_command
