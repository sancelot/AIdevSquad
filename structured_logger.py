# File: structured_logger.py

import os
import datetime
import json
import datetime 
class StructuredLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.log_dir, f"run_trace_{timestamp}.md")
        self.json_log_path = os.path.join(self.log_dir, f"run_trace_{timestamp}.json")

        self.entries = []
        self.current_task = None
        self.current_step = None
        self.start_time = datetime.datetime.now(datetime.timezone.utc) # Record the start time of the run

        
        print(f"📝 Execution trace will be saved to: {self.json_log_path}")

    def _flush(self):
        """Internal method to write the current log entries to disk."""
        try:
            # Write JSON log
            with open(self.json_log_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            
            # Write Markdown log
            # The logic to generate markdown can be moved into a helper function
            # for clarity, but for now, we can just call the write logic.
            self._write_markdown_file()

        except Exception as e:
            print(f"⚠️ Logger Warning: Failed to flush logs to disk. Error: {e}",self.entries)

    def log(self, level, message):
        """Adds a general, non-nested log entry."""
        entry = {"type": "general", "level": level.upper(), "message": message}
        self.entries.append(entry)
        print(f"[{level.upper()}] {message}") 
        self._flush()

    def log_phase(self, phase_name):
        """Logs the start of a major phase."""
        entry = self._create_entry("phase", {"name": phase_name})
        self.entries.append(entry)
        self.current_task = None # Reset task context when a new phase begins
        print("\n" + "="*60)
        print(f"PHASE START: {phase_name.upper()}")
        print("="*60)
        self._flush()

    def log_task(self, task_number, total_tasks, task_description):
        """Logs the start of a new task and sets it as the current context."""
        entry = self._create_entry("task", {
            "number": task_number,
            "total": total_tasks,
            "description": task_description,
            "steps": []
        })
        self.entries.append(entry)
        self.current_task = entry
        self.current_step = None # Reset step context
        print(f"\n--- 🚀 Starting Task {task_number}/{total_tasks}: {task_description} ---")
        self._flush()

    def start_step(self, step_number, max_steps):
        """Starts a new step within the current task."""
        if self.current_task is None:
            print("⚠️ Logger Warning: start_step called without an active task.")
            return
        step_entry = self._create_entry("step", {
            "number": step_number,
            "max": max_steps,
            "events": []
        })
        # Safely access 'steps' key
        if "steps" not in self.current_task:
            self.current_task["steps"] = []
        self.current_task["steps"].append(step_entry)
        self.current_step = step_entry
        print(f"--- Action Step {step_number}/{max_steps} ---")
        self._flush()

    def _create_entry(self, entry_type, data):
        """Internal helper to create a timestamped log entry."""
        # Get current time in UTC and format it cleanly
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # The output will be like '2023-10-27T14:55:00.123456+00:00'
        # To make it 'Z' format:
        timestamp_str = timestamp_str.replace('+00:00', 'Z')
        return {
            "timestamp":timestamp_str,
            "type": entry_type,
            **data
        }
    def log_event_in_step(self, event_type, data):
        """
        Logs a specific event (e.g., llm_call, thought) within the current step.
        'data' is a dictionary containing the event's payload.
        """
        if self.current_step is None:
            print(f"⚠️ Logger Warning: log_event_in_step('{event_type}') called without an active step.")
            return
        event = self._create_entry(event_type, data)
        self.current_step["events"].append(event)
        
        # Optional: Print to console for real-time feedback
        if event_type == "thought":
            print(f"🧠 Agent's Thought: {data.get('thought', 'N/A')}")
        elif event_type == "tool_call":
            print(f"▶️ Calling Tool: `{data.get('tool_name')}` with args: `{data.get('tool_args')}`")
        elif event_type == "tool_result":
            result_str = str(data.get('result', 'N/A'))
            if len(result_str) > 300:
                result_str = result_str[:300] + " ... (truncated)"
            print(f"🛠️ Tool Result: {result_str}")
        self._flush()

    def log_final_summary(self, cost_summary, metrics_summary):
        """Logs the final summary information."""
        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = end_time - self.start_time
        
        final_summary_entry = self._create_entry("final_summary", {
            "start_time": self.start_time.isoformat().replace('+00:00', 'Z'),
            "end_time": end_time.isoformat().replace('+00:00', 'Z'),
            "total_duration_seconds": duration.total_seconds(),
            "cost_summary_text": cost_summary,
            "metrics_summary_text": metrics_summary,
        })
        self.entries.append(final_summary_entry)
        print("Finalizing log files...")
        self._flush()

    # The final write_to_file is now just a final flush
    def write_to_file(self):
        self._flush()

    def _write_markdown_file(self):
        # --- Write Markdown log ---
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write("# Agent Execution Trace\n")
            for entry in self.entries:
                if entry["type"] == "phase":
                    f.write(f"\n## PHASE: {entry['name'].upper()}\n")
                elif entry["type"] == "general":
                    f.write(f"\n*[{entry['level']}] {entry['message']}*\n")
                elif entry["type"] == "task":
                    f.write(f"\n### Task {entry['number']}/{entry['total']}: {entry['description']}\n")
                    if "steps" in entry:
                        for step in entry["steps"]:
                            f.write(f"\n#### Step {step['number']}/{step['max']}\n")
                            for event in step["events"]:
                                
                                if event["type"] == "llm_call":
                                    f.write(f"**Agent:** `{event.get('agent_name', 'N/A')}`\n")
                                    f.write("**Prompt sent to LLM:**\n")
                                    f.write(f"```\n{event.get('prompt', '')}\n```\n")
                                elif event["type"] == "llm_response":
                                    f.write("**Raw response from LLM:**\n")
                                    f.write(f"```json\n{event.get('response', '')}\n```\n")
                                elif event["type"] == "thought":
                                     f.write(f"**Thought:** *{event.get('thought', '')}*\n\n")
                                elif event["type"] == "tool_call":
                                    f.write(f"**Action:** Calling tool `{event.get('tool_name')}` with arguments:\n")
                                    f.write(f"```json\n{json.dumps(event.get('tool_args', {}), indent=2)}\n```\n")
                                elif event["type"] == "tool_result":
                                    f.write("**Observation (Tool Result):**\n")
                                    f.write(f"```\n{str(event.get('result', ''))}\n```\n---\n")