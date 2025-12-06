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
        team_manifest = """
        AVAILABLE TEAMMATES:
        1. ProductOwnerAgent: Holds the master plan. Consult them if logic is ambiguous or if you need to cut scope.
        2. CodeReviewerAgent: Expert in security and patterns. Consult them *early* for architectural advice, not just at the end.
        """
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
                                       description="Parse a file (AST for Python) and return high-level structure: classes, functions with arguments, imports, and global variables. Use to understand file structure without reading full content."),
            FunctionTool.from_defaults(
                fn=orchestrator_tools.call_agent,
                name="collaborate_with_teammate",  # New semantic name
                description=f"""Send a message to a teammate to discuss the task, ask for advice, or request a review.

                This is a chat interface. You can send code snippets, error logs, or implementation plans.

                {team_manifest}

                Returns: The teammate's response (which may include suggested code or feedback).
                """
            )
        ]

        self.logger = logger
        system_message = """You are a Senior Software Engineer working in a high-performance AI team. 
Your goal is not just to write code, but to ship a robust, agreed-upon solution.

**YOUR WORKING STYLE:**
1.  **You are NOT a solo worker.** You are part of a swarm. Complex problems require discussion.
2.  **Iterative Development:** Don't try to write the whole file in one shot. Write the skeleton, **ask the Reviewer if the structure looks right**, then fill in the logic.
3.  **Context Switching:** It is acceptable to pause coding to clarify a requirement with the ProductOwner, then return to coding.

**WHEN TO COLLABORATE:**
-   **Before Coding:** If the task is vague, ask the `ProductOwnerAgent`.
-   **During Coding:** If you are touching critical legacy code, ask the `CodeReviewerAgent` "I am planning to change X to Y, does this break anything?"
-   **After Coding:** Ask the `CodeReviewerAgent` to verify your specific changes.

**THE LOOP:**
1.  **Plan:** Read files and map out changes.
2.  **Consult (Optional but recommended):** If the plan is complex, run it by a teammate via `collaborate_with_teammate`.
3.  **Execute:** Use your coding tools.
4.  **Verify:** Check your work.
5.  **Sign-off:** You cannot consider the task done until you are confident the changes meet the team's standards.

**Refrain from guessing.** If a variable name or logic is unclear, asking a teammate is cheaper than rewriting it later.
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

    async def chat(self, message: str) -> str:
        """
        Handle a chat message from another agent (e.g. ProductOwnerAgent, CodeReviewerAgent).
        We create a temporary ReActAgent to allow tool usage (reading files, checking code) 
        without full task execution overhead.
        """
        chat_system_message = """You are a Senior Software Engineer being consulted by a teammate.
        You have access to all your development tools (reading files, listing files, etc.).
        
        Answer the teammate's question directly and helpfully.
        You can use tools to check the code or project state if needed to answer accurately.
        """

        # Create a temporary agent for this interaction
        # We reuse the same tools, but in a new agent instance for this specific conversation context
        chat_agent = ReActAgent(
            llm=self.llm,
            tools=self.agent.tools,  # Reuse the full toolset
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
                            "nested_agent": "DeveloperAgent(Chat)"
                        }
                    )
                    self.logger.log_event_in_step(
                        "tool_result", {
                            "result": ev.tool_output.content,
                            "nested_agent": "DeveloperAgent(Chat)"
                        }
                    )
                elif isinstance(ev, AgentOutput):
                    final_answer_str = str(ev.response)
                elif isinstance(ev, AgentStream):
                    # Accumulate stream if needed, or just let it flow
                    pass

            self.logger.log_event_in_step(
                "llm_response", {
                    "response": final_answer_str,
                    "nested_agent": "DeveloperAgent(Chat)"
                }
            )
            return final_answer_str

        except Exception as e:
            error_msg = f"Error during DeveloperAgent chat: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg
