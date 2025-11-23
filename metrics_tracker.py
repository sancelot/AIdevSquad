# File: metrics_tracker.py

import json

class MetricsTracker:
    def __init__(self):
        # Raw data storage
        self.tasks = {} # {task_description: {"status": "success/fail", "review_cycles": 2, "churn": 150}}
        self.tool_usage = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "calls_by_tool": {} # e.g., {"read_file": {"success": 5, "fail": 0}}
        }
    
    def track_tool_call(self, tool_name, success):
        """Records the usage of a tool and its outcome."""
        self.tool_usage["total_calls"] += 1
        
        # Initialize counter for a new tool
        if tool_name not in self.tool_usage["calls_by_tool"]:
            self.tool_usage["calls_by_tool"][tool_name] = {"success": 0, "fail": 0}

        if success:
            self.tool_usage["successful_calls"] += 1
            self.tool_usage["calls_by_tool"][tool_name]["success"] += 1
        else:
            self.tool_usage["failed_calls"] += 1
            self.tool_usage["calls_by_tool"][tool_name]["fail"] += 1

    def start_task(self, task_description):
        """Records the start of a new task."""
        if task_description not in self.tasks:
            self.tasks[task_description] = {
                "status": "in_progress",
                "review_cycles": 0,
                "code_churn": 0,
                "passed_first_review": None # None=not reviewed, True=passed, False=failed
            }

    def increment_review_cycle(self, task_description):
        """Increments the review cycle count for a task."""
        if task_description in self.tasks:
            self.tasks[task_description]["review_cycles"] += 1
            if self.tasks[task_description]["passed_first_review"] is None:
                # If we are entering a second review cycle, it means the first one failed
                self.tasks[task_description]["passed_first_review"] = False

    def add_code_churn(self, task_description, diff_size):
        """Adds the number of lines changed during a fix to the task's churn count."""
        if task_description in self.tasks:
            self.tasks[task_description]["code_churn"] += diff_size

    def complete_task(self, task_description, success, review_passed_on_first_try):
        """Marks a task as completed, either successfully or as a failure."""
        if task_description in self.tasks:
            self.tasks[task_description]["status"] = "success" if success else "fail"
            if self.tasks[task_description]["passed_first_review"] is None:
                self.tasks[task_description]["passed_first_review"] = review_passed_on_first_try

    def get_summary(self):
        """Calculates and returns a formatted summary of all metrics."""
        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return "No tasks were attempted."

        # 1. Success Rate
        successful_tasks = sum(1 for t in self.tasks.values() if t["status"] == "success")
        success_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0

        # 2. Review Cycles
        reviewed_tasks = [t for t in self.tasks.values() if t["review_cycles"] > 0]
        total_review_cycles = sum(t["review_cycles"] for t in reviewed_tasks)
        avg_review_cycles = total_review_cycles / len(reviewed_tasks) if reviewed_tasks else 0

        # 3. Code Preservation (Churn)
        total_churn = sum(t["code_churn"] for t in self.tasks.values())

        # 4. First-Time Pass Rate
        tasks_that_went_to_review = [t for t in self.tasks.values() if t["passed_first_review"] is not None]
        passed_on_first_try = sum(1 for t in tasks_that_went_to_review if t["passed_first_review"] is True)
        first_time_pass_rate = (passed_on_first_try / len(tasks_that_went_to_review)) * 100 if tasks_that_went_to_review else 100

        summary = "\n" + "="*50 + "\n"
        summary += "📈 PERFORMANCE METRICS REPORT 📈\n"
        summary += "="*50 + "\n"
        summary += f"Overall Success Rate      : {success_rate:.2f}% ({successful_tasks}/{total_tasks} tasks)\n"
        summary += f"First-Time Pass Rate      : {first_time_pass_rate:.2f}% ({passed_on_first_try}/{len(tasks_that_went_to_review)} reviewed tasks)\n"
        summary += f"Average Review Cycles     : {avg_review_cycles:.2f} per reviewed task\n"
        summary += f"Total Code Churn (lines)  : {total_churn} lines changed during fixes\n"

        summary += "\n" + "-"*50 + "\n"
        summary += "🛠️ TOOL USAGE STATISTICS 🛠️\n"
        summary += "-"*50 + "\n"
        summary += f"Total Tool Calls          : {self.tool_usage['total_calls']}\n"
        summary += f"  - Successful            : {self.tool_usage['successful_calls']}\n"
        summary += f"  - Failed                : {self.tool_usage['failed_calls']}\n\n"
        
        summary += "Breakdown by Tool:\n"
        for tool_name, stats in self.tool_usage["calls_by_tool"].items():
            total = stats['success'] + stats['fail']
            fail_rate = (stats['fail'] / total) * 100 if total > 0 else 0
            summary += f"  - {tool_name:<20}: {total:<4} calls ({stats['success']} success, {stats['fail']} fail, {fail_rate:.1f}% fail rate)\n"
        # -----------
        summary += "="*50 + "\n"
        return summary

    def to_dict(self):
        """Exports state for saving."""
        return {"tasks": self.tasks,"tool_usage": self.tool_usage}

    def from_dict(self, data):
        """Loads state from a dictionary."""
        self.tasks = data.get("tasks", {})
        self.tool_usage = data.get("tool_usage", self.tool_usage)