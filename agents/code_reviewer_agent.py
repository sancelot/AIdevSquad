import asyncio
import traceback
from .base_agent import BaseAgent
from .utils import parse_json_from_response
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.workflow import Context, StopEvent
from llama_index.core.agent.workflow import AgentStream, ToolCallResult, AgentOutput


class CodeReviewerAgent(BaseAgent):
    def __init__(self, llm_class, llm_args, orchestrator_tools, logger):
        super().__init__(llm_class, llm_args, temperature=0.1, agent_name="CodeReviewerAgent")
        self.logger = logger
        # CRITICAL: Provide READ-ONLY tools.
        read_only_tools = [
            FunctionTool.from_defaults(
                fn=orchestrator_tools.read_file,
                name="read_file",
                description="Read the full content of a specific file to understand its context and functionality."
            ),
            FunctionTool.from_defaults(
                fn=orchestrator_tools.list_files,
                name="list_files",
                description="List all files in the project to understand the overall structure."
            ),
            FunctionTool.from_defaults(
                fn=orchestrator_tools.get_code_summary,
                name="get_code_summary",
                description="Get a high-level summary (imports, classes, functions) of a Python file. Useful for quickly understanding a file's role without reading all its code."
            ),
            FunctionTool.from_defaults(
                fn=orchestrator_tools.submit_review,
                name="submit_review",
                description="""Call this SINGLE tool when your investigation is complete to submit your final report."""
            )
        ]
        system_message = """You are an expert code reviewer AI. Your job is to investigate code changes and report issues.

**CRITICAL RULE: YOU MUST ALWAYS CALL THE submit_review TOOL TO FINISH.**
Never "answer" conversationally. Never say "No issues found" in text. ALWAYS call submit_review() to complete your task.

**INVESTIGATION PROCESS:**
1. **Analyze:** Read the task and git diff to understand what changed
2. **Investigate:** Use read_file, list_files, get_code_summary to gather context
3. **Evaluate:** Check for:
   - Missing imports or dependencies
   - Broken integrations (routes not linked, functions not called)
   - Logic errors or incomplete implementations
   - Style issues or typos
4. **MANDATORY: Call submit_review** - This is the ONLY way to complete your task

**COMPLETION RULES:**
✓ Code is perfect? → Call submit_review([])
✓ Found issues? → Call submit_review([issue objects])
✗ NEVER say "No issues found" as text
✗ NEVER try to "answer" without calling submit_review
✗ NEVER finish without calling submit_review

**ISSUE STRUCTURE:**
Each issue dictionary must have:
- severity: "critical", "major", "minor", or "suggestion"
- type: "typo", "style_violation", "import_error", "naming_convention", 
        "missing_feature", "incomplete_feature", "logic_bug", "integration_issue"
- file: Full file path
- line: Line number (approximate)
- description: Clear one-sentence problem description
- suggestion: Simple fix recommendation


**Example 1 - Issues found**
*Final thought:* "My investigation is complete. I will now call `submit_review` to submit my report."
*Tool Call:* `submit_review(report={"issues": [{"severity": "major", "type": "integration_issue", "file": "app.py", "line": 25, "description": "...", "suggestion": "..."}]})

**Example 2 - No issues**
*Final thought:* "My investigation is complete. I will now call `submit_review` to submit my report."
*Tool Call:* `submit_review(report={"issues": []})`

"""
        self.streaming = True
        self.agent = ReActAgent(
            llm=self.llm,
            tools=read_only_tools,
            system_prompt=system_message,
            verbose=False,  # Good for debugging the reviewer's thought process
            streaming=self.streaming  # Enable streaming to get events

        )
    # Added 'task' argument

    async def review_code(self, task, project_structure, git_diff, previous_issues=None):
        """
         Triggers the ReAct agent to review code changes.
         """
        # The prompt is now an initial briefing for the investigator, not the full context.
        prompt = f"""
        ** Mission Briefing: **

        **Task to be Verified: **
        {task}
        ** Current Project File Structure: **
        {project_structure}
        ** Initial Evidence(Code Changes to Review): ** ```diff
        {git_diff}
        ```
        """
        if previous_issues:
            prompt += f"\n**Secondary Objective:** You are reviewing a fix. Verify if these previous issues were resolved: {previous_issues}."
        prompt += """**YOUR INSTRUCTIONS:**
1. Investigate using your tools (read_file, list_files, etc.)
2. When investigation is complete, you MUST call submit_review() tool
3. If code is perfect: submit_review([])
4. If you found issues: submit_review([...issue objects...])

DO NOT write "No issues found" or "The file was created successfully" as your answer.
DO NOT try to answer conversationally.
YOU MUST CALL submit_review to complete this task.

Begin your investigation now."""
        print("---------------------")  # FIXME debug msg
        print("code reviw prompt:")
        print(prompt)
        print("-----------------")
        tool_usage_history = []
        final_issues = None

        ctx = Context(self.agent)
        handler = self.agent.run(prompt, ctx=ctx)
        try:
            event_count = 0
            async for ev in handler.stream_events():
                event_count += 1
                if isinstance(ev, ToolCallResult):
                    print("toolcall result")
                    # Log for the orchestrator
                    tool_usage_history.append({
                        "tool_name": ev.tool_name,
                        "arguments": ev.tool_kwargs,
                        "result": str(ev.tool_output)
                    })
                    # Log for real-time console view
                    # Truncate long outputs
                    print(
                        f"\n[Reviewer] Call {ev.tool_name} with {ev.tool_kwargs}\n[Reviewer] Returned: {str(ev.tool_output)[:200]}...")
                    if ev.tool_name == 'submit_review':
                        # The arguments to the tool ARE our final list of issues
                        report = ev.tool_kwargs.get('report', None)
                        final_issues = report.get('issues', [])
                        self.logger.log("INFO", f"FIXME submit_review")
                        self.logger.log("INFO", f"{report}")
                        self.logger.log("INFO", f"final issues {final_issues}")

                        self.logger.log(
                            "INFO", f"result {str(ev.tool_output)}")
                        print(
                            f"\n[Reviewer] Final report submitted via submit_review tool.")
                if isinstance(ev, StopEvent):
                    print("DEBUG: Got StopEvent, breaking...")
                    break
                if isinstance(ev, AgentOutput):
                    print("DEBUG: Got AgentOutput event...")
                    final_answer_str = str(ev.response)
                if isinstance(ev, AgentStream):
                    print(f"{ev.delta}", end="", flush=True)

            print(f"DEBUG: Event stream finished. Total events: {event_count}")

            print("DEBUG: Handler completed successfully")
        except asyncio.TimeoutError:
            print("ERROR: Handler timed out after 5 seconds - likely stuck waiting")
            print(
                f"DEBUG: Runtime state - receive_queue size: {ctx.runtime.receive_queue.qsize()}")
            print(
                f"DEBUG: Runtime state - publish_queue size: {ctx.runtime.publish_queue.qsize()}")
            traceback.print_exc()
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            final_answer = None

        self.logger.log_event_in_step(
            "llm_response", {"response": final_answer_str})
        print(final_issues, type(final_issues))
        if final_issues is not None and isinstance(final_issues, list):
            return final_issues, tool_usage_history
        else:
            # Failure case: Parsing failed, or the JSON structure was wrong.
            print(
                "❌ Error: CodeReviewerAgent failed to return a valid list of issues.")
            error_issue = [{"severity": "critical", "type": "parsing_error", "file": "N/A", "line": 0,
                            "description": "CodeReviewerAgent failed to produce a valid response.", "suggestion": "The LLM response was malformed or missing the 'issues' key."}]
            return error_issue, tool_usage_history
        # ----------------------------------------

    async def chat(self, message: str) -> str:
        """
        Handle a chat message from another agent (e.g. DeveloperAgent).
        We create a temporary ReActAgent to allow tool usage (reading files) 
        without the strict 'submit_review' constraint of the main review loop.
        """
        chat_system_message = """You are an expert code reviewer. You are being consulted by a developer.
        You have access to tools to read files and analyze code.
        Answer the developer's question directly and helpfully.
        Do NOT call `submit_review`. Just answer in text.
        """

        # Create a temporary agent for this interaction
        chat_agent = ReActAgent(
            llm=self.llm,
            tools=self.agent.tools,  # Reuse the read-only tools
            system_prompt=chat_system_message,
            verbose=False,
            streaming=True
        )

        ctx = Context(chat_agent)
        handler = chat_agent.run(message, ctx=ctx)
        final_answer_str = ""

        try:
            async for ev in handler.stream_events():
                if isinstance(ev, ToolCallResult):
                    # Log nested tool calls
                    self.logger.log_event_in_step(
                        "tool_call", {
                            "tool_name": ev.tool_name,
                            "tool_args": ev.tool_kwargs,
                            "nested_agent": "CodeReviewerAgent(Chat)"
                        }
                    )
                    self.logger.log_event_in_step(
                        "tool_result", {
                            "result": ev.tool_output.content,
                            "nested_agent": "CodeReviewerAgent(Chat)"
                        }
                    )
                elif isinstance(ev, AgentOutput):
                    final_answer_str = str(ev.response)

            self.logger.log_event_in_step(
                "llm_response", {
                    "response": final_answer_str,
                    "nested_agent": "CodeReviewerAgent(Chat)"
                }
            )
            return final_answer_str

        except Exception as e:
            error_msg = f"Error during CodeReviewerAgent chat: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg
