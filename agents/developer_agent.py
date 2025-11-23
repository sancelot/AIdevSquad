import asyncio
import traceback
from .utils import parse_json_from_response
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from .base_agent import BaseAgent
import sys
from llama_index.core.workflow import Context, StopEvent
from llama_index.core.agent.workflow import AgentStream, ToolCallResult, AgentOutput


class DeveloperAgent(BaseAgent):
    def __init__(self, llm_class, llm_args, orchestrator_tools, logger):
        super().__init__(llm_class, llm_args, temperature=0.1, agent_name="DeveloperAgent")
        tools = [
            # === SEMANTIC REFACTORING TOOLS (PREFERRED) ===
            FunctionTool.from_defaults(
                fn=orchestrator_tools.add_code_block,
                name="add_code_block",
                description="""Add a code block (function, class, method) at a semantic location.

This tool is PREFERRED over text-based insertion for adding complete, valid blocks of code.

Key Features:
-   Uses AST for semantic placement (e.g., after a specific function).
-   Automatically handles indentation.
-   Validates syntax before applying the change.

Example 1: Adding a simple function after another.
  add_code_block(
      filepath="app.py",
      new_code="def new_function():\\n    print('Hello')",
      location="after_function",
      target_name="existing_function"
  )

Example 2: Adding a new Flask route at the end of the file.
  add_code_block(
      filepath="app.py",
      new_code="@app.route('/new_route')\\ndef handle_new_route():\\n    return 'Success'",
      location="end_of_file"
  )

Locations:
  - "end_of_file": At the end of the file.
  - "beginning_of_file": After all imports.
  - "inside_class": Inside a class (requires 'target_name').
  - "after_function" or "before_function": Relative to a function (requires 'target_name').
  - "after_class" or "before_class": Relative to a class (requires 'target_name').

Returns: {"success": bool, "message": str, "error": str (if failed)}"""
            ),

            FunctionTool.from_defaults(
                fn=orchestrator_tools.refactor_rename_symbol,
                name="refactor_rename_symbol",
                description="""Safely rename a symbol (function, class, variable, interface) using AST.

PREFERRED over replace_text for renaming because:
- Only renames actual code (not comments/strings)
- Handles all occurrences safely
- Validates new name
- No risk of partial matches

Usage:
  refactor_rename_symbol(
      filepath="app.py",
      old_name="calc_ttl",
      new_name="calculate_total",
      symbol_type="function"  # "function", "class", "variable", "interface", "all"
  )

Symbol Types:
  - "function" - Only rename functions
  - "class" - Only rename classes
  - "variable" - Only rename variables
  - "interface" - Only rename interfaces (TypeScript)
  - "all" - Rename all occurrences (use with caution!)

Returns: {"success": bool, "occurrences": int, "locations": list}

Example: Rename a function based on code review feedback."""
            ),

            FunctionTool.from_defaults(
                fn=orchestrator_tools.delete_code_block,
                name="delete_code_block",
                description="""Delete a code block (function, class, interface) by name.

PREFERRED over replace_text for deletion because:
- Just give the name, no need to copy entire code
- Handles multi-line blocks automatically
- Removes decorators/comments too
- Validates result

Usage:
  delete_code_block(
      filepath="app.py",
      block_name="old_api_call",
      block_type="function"  # "function", "class", "interface", "auto"
  )

Block Types:
  - "auto" - Auto-detect type
  - "function" - Only delete functions
  - "class" - Only delete classes
  - "interface" - Only delete interfaces (TypeScript)

Returns: {"success": bool, "lines_deleted": int}

Example: Remove obsolete code without manual text matching."""
            ),
            FunctionTool.from_defaults(fn=orchestrator_tools.replace_text),
            FunctionTool.from_defaults(fn=orchestrator_tools.insert_text),
            FunctionTool.from_defaults(
                fn=orchestrator_tools.read_file, description="Read a file"),
            FunctionTool.from_defaults(fn=orchestrator_tools.write_file,
                                       description="Create a NEW file with content. NEVER use for modifying existing files."),
            FunctionTool.from_defaults(fn=orchestrator_tools.list_files,
                                       description="List all files in the project or a directory. Use this to explore project structure."),
            FunctionTool.from_defaults(
                fn=orchestrator_tools.create_directory, description="Create a new directory in the project."),
            FunctionTool.from_defaults(
                fn=orchestrator_tools.delete_file,
                name="delete_file",
                description="""Deletes a specified file from the project workspace.
                Use this to clean up unnecessary files, remove files created by mistake, or resolve issues with leftover code.
                This operation is PERMANENT and cannot be undone.
                Args:
                    - filename (str): The path of the file to delete, relative to the project root.
                Returns a dictionary indicating success or failure.
                """
            ),
            FunctionTool.from_defaults(fn=orchestrator_tools.get_code_summary,
                                       description="Parse a file (AST for Python) and return high-level structure: classes, functions with arguments, imports, and global variables. Use to understand file structure without reading full content.")

            # ... add all other tools
        ]

        self.logger = logger
        system_message = """You are an expert, surgical AI Software Engineer. Your goal is to solve the user's task by making precise, minimal changes to the codebase by thinking step-by-step and using tools.

**--- CORE DIRECTIVES ---**
1.  **FOCUS ON THE CURRENT TASK:** Your goal is to execute ONLY the single task described in the "YOUR CURRENT ASSIGNMENT" section of the user prompt.
2.  **DO NOT JUMP AHEAD:** Do not work on "Upcoming Tasks" or any task other than the one currently assigned.
3.  **USE REFERENCE DOCUMENTS:** The user prompt will contain reference documents (Mission, Blueprint, Plan). Use them to understand the context and ensure your work aligns with the project goals, but do not act on them directly.
4.  **BE SURGICAL:** NEVER use `write_file` to modify an existing file. `write_file` is ONLY for creating NEW files. To modify a file, you MUST use `read_file` first, then a precise tool like `insert_text` or `replace_text`.
5.  **PLAN COMPLEX TASKS:** If a task involves multiple changes, create a short internal checklist in your 'Thought' and execute it step-by-step.
6.  **VERIFY YOUR WORK:** Before finishing, reflect on your changes. Does the code fully complete the CURRENT TASK? Are all dependencies (imports, routes) met?


**REASONING PROCESS (Read -> Plan -> Act -> Verify):**
1.  **Read:** If you need to know the state of a file, use `read_file`.
2.  **Plan:** Formulate a plan, including a checklist if needed.
3.  **Act:** Execute your plan by calling the appropriate tool.
4.  **Verify:** After acting, check your work and its dependencies. If more changes are needed, go back to step 2.
5.  **Finish:** When the task is truly complete and verified, your final thought should be "The task is complete," and you should not call any more tools.

**FORMATTING CODE WITHIN JSON STRINGS:** When you provide a block of code (Python, HTML, JSON, etc.) as a string value in an argument, you MUST ensure it is a valid JSON string literal. This means:
    *   Use `\n` for all newlines.
    *   Escape all double quotes (`"`) inside the code with a backslash (`\"`).
    *   Escape all backslashes (`\`) inside the code with another backslash (`\\`).

"""
        self.streaming = True
        self.agent = ReActAgent(llm=self.llm, verbose=False,
                                system_prompt=system_message, tools=tools, streaming=self.streaming)

    async def execute_task(self, task, rag_context=""):

        # Create a context to store the conversation history/session state
        self.logger.log("INFO", f"DeveloperAgent starting task: {task}")
        self.logger.log_event_in_step(
            "llm_call", {"agent_name": "DeveloperAgent", "prompt": task})

        self.logger.log_event_in_step(
            "llm_call", {"agent_name": "DeveloperAgent", "prompt": task})
        ctx = Context(self.agent)
        handler = self.agent.run(task, ctx=ctx)
        tool_calls = []

        if self.streaming:
            s = ""
            try:
                event_count = 0
                async for ev in handler.stream_events():
                    if isinstance(ev, ToolCallResult):
                        print(
                            f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
                        self.logger.log_event_in_step(
                            "tool_call", {"tool_name": ev.tool_name, "tool_args": ev.tool_kwargs})
                        self.logger.log_event_in_step(
                            "tool_result", {"result": ev.tool_output.content})
                        tool_calls.append({"tool_name": ev.tool_name,
                                           "tool_args": ev.tool_kwargs,
                                           })
                    if isinstance(ev, StopEvent):
                        print("DEBUG: Got StopEvent, breaking...")
                        break
                    if isinstance(ev, AgentOutput):
                        final_answer_str = str(ev.response)
                    if isinstance(ev, AgentStream):
                        print(f"{ev.delta}", end="", flush=True)
                        s += ev.delta

                print(
                    f"DEBUG: Event stream finished. Total events: {event_count}")
                print("DEBUG: About to await handler...")

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
        else:
            # no streaming mode
            try:
                # Simple await sans streaming
                final_answer = await asyncio.wait_for(handler, timeout=600.0)

                print("DEBUG: Agent completed successfully")

                # Afficher la réponse
                final_answer_str = str(final_answer)
                print("=================final answer============")
                print(final_answer_str)
                print("------------")
                print([attr for attr in dir(final_answer)
                      if not attr.startswith('__')])
                # Essayer d'accéder aux tool calls
                if hasattr(final_answer, 'tool_calls'):
                    print(f"\nTool calls used: {final_answer.tool_calls}")
                    try:
                        for tool in final_answer.tool_calls:
                            print(tool)
                            print(tool.tool_name)
                            print(tool.tool_kwargs)
                            print(tool.tool_output)
                            self.logger.log_event_in_step(
                                "tool_call", {"tool_name": tool.tool_name, "tool_args": tool.tool_kwargs})
                            self.logger.log_event_in_step(
                                "tool_result", {"result": tool.tool_output.content})
                            tool_calls.append({"tool_name": tool.tool_name,
                                               "tool_args": tool.tool_kwargs,
                                               })
                    except Exception as e:
                        print("eceprion ", e)
                if hasattr(final_answer, 'metadata'):
                    print(f"\nMetadata: {final_answer.metadata}")

                self.logger.log_event_in_step(
                    "final_answer", {"agent": "DeveloperAgent", "answer": final_answer_str})

            except asyncio.TimeoutError:
                print("ERROR: Agent timed out after 300 seconds")
                return False, None
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                return False, None

        print("=================final answer============")
        print(final_answer_str)
        print("------------")

        # # Check handler attributes
        # if hasattr(handler, 'tool_calls'):
        #     print(f"Tool calls: {handler.tool_calls}")
        # print("----")
        # print(handler)
        # # Check context
        # if hasattr(ctx, 'get_tool_calls'):
        #     print(f"Tool calls from context: {ctx.get_tool_calls()}")

        self.logger.log_event_in_step(
            "llm_response", {"response": final_answer_str})

        return True, final_answer_str, tool_calls
