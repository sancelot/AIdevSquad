from orchestrator_tools import OrchestratorTools
from structured_logger import StructuredLogger
from metrics_tracker import MetricsTracker
from config import debug
import git
import os
import subprocess
import docker
from CostTracker import CostTracker
from dotenv import load_dotenv
from agents import (CodeReviewerAgent, DocumentationAgent, ProductOwnerAgent, DeveloperAgent,
                    RequirementsAnalystAgent, SoftwareArchitectAgent, TesterAgent, UnitTestAgent, configure_llm_and_embed, SADTSARTPlannerAgent)
import shutil
import json
from llama_index.core import VectorStoreIndex, Document
import ast
import sys

load_dotenv()

try:
    from google.genai.errors import ClientError
except ImportError:
    ClientError = None  # In case you run with ollama and don't have the lib installed


class Orchestrator:
    def __init__(self, project_name, initial_prompt_file=None, provider="ollama", force_new=False, run_tests=False):

        # --- 1. CORE SETUP ---
        self.logger = StructuredLogger()
        llm_class, llm_args, token_counter = configure_llm_and_embed(provider)
        self.model_name = llm_args.get("model_name") or llm_args.get("model")

        self.cost_tracker = CostTracker(token_counter)

        # --- 2. ATTRIBUTE INITIALIZATION (with defaults) ---
        self.project_name = project_name
        self.project_path = os.path.join(".", "projects", project_name)
        self.requirements_file = os.path.join(
            self.project_path, "requirements.md")
        self.workspace_dir = os.path.join(self.project_path, "workspace")
        self.state_file = os.path.join(self.project_path, "state.json")

        self.project_name = project_name
        self.project_dir = self.workspace_dir
        self.repo = None
        self.run_tests_enabled = run_tests

        self.plan = []
        self.files = {}  # Dictionary { 'filename': 'content' }
        self.run_command = ""
        self.last_completed_task_index = -1
        self.code_index = None

        self.metrics_tracker = MetricsTracker()
        # --- 3. STATE MANAGEMENT (Load or Start New) ---

        # Prompt loading is now dynamic
        if os.path.exists(self.requirements_file) and not force_new:
            with open(self.requirements_file, 'r', encoding='utf-8') as f:
                self.user_prompt = f.read()  # Load the entire file as context
        elif initial_prompt_file:
            with open(initial_prompt_file, 'r', encoding='utf-8') as f:
                self.user_prompt = f.read()
        else:
            raise ValueError(
                "No requirements file found and no initial prompt file provided.")

        if os.path.exists(self.state_file) and not force_new:
            self.logger.log("INFO", "Resuming existing project...")
            self._load_state()
            self.repo = git.Repo(self.project_dir)
            # IMPORTANT : Il faut construire l'index au démarrage à partir des fichiers existants
            self._build_index_from_disk()
        else:
            self.logger.log("INFO", "Starting a new project...")
            self._clean_workspace()

        # --- 4. AGENT INITIALIZATION ---
        # Pass self so tools can access self.files
        tools = OrchestratorTools(self)
        self.po_agent = ProductOwnerAgent(llm_class, llm_args)
        self.dev_agent = DeveloperAgent(
            llm_class, llm_args, tools, self.logger)
        self.reviewer_agent = CodeReviewerAgent(
            llm_class, llm_args, tools, self.logger)
        self.doc_agent = DocumentationAgent(llm_class, llm_args)
        # self.tester_agent = TesterAgent(llm_class,llm_args,self.project_dir)
        self.tester_agent = None
        self.unit_test_agent = UnitTestAgent(llm_class, llm_args)
        self.requirements_agent = RequirementsAnalystAgent(llm_class, llm_args)
        self.architect_agent = SoftwareArchitectAgent(llm_class, llm_args)
        self.sadt_sart_agent = SADTSARTPlannerAgent(llm_class, llm_args)

    def _get_project_diff_since_last_commit(self):
        """Returns a git diff of all staged and unstaged changes against the last commit."""
        if not self.repo:
            return "Git repository not initialized."
        # Add all files to the index (staging area) so that new, untracked files are included in the diff.
        # This does NOT commit them.
        self.repo.git.add(A=True)

        # Now, get the diff of the staged changes against the last commit.
        return self.repo.git.diff('HEAD')

    def _git_commit(self, message):
        """Adds all changes and creates a commit with the given message."""
        try:
            if self.repo.is_dirty(untracked_files=True):
                self.repo.git.add(A=True)
                self.repo.index.commit(message)
                print(f"✅ Git commit created: '{message}'")
        except Exception as e:
            print(f"⚠️ Git commit failed: {e}")
            sys.exit(1)

    def _get_project_diff(self):
        """Returns a git diff of all staged and unstaged changes."""
        if not self.repo:
            return "Git repository not initialized."

        # We add all changes to the index to get a complete diff
        self.repo.git.add(A=True)
        # Get the diff against the last commit (HEAD)
        return self.repo.git.diff('HEAD')

    def _validate_file_path(self, filename):
        """
        Validate file path works on Windows, Linux, and macOS.
        Prevents path traversal attacks.
        """
        if not filename:
            raise ValueError("Filename cannot be empty")

        # Step 1: Normalize the path (handles / and \ on all platforms)
        normalized = os.path.normpath(filename)

        # Step 2: Check for absolute paths or parent directory references
        if os.path.isabs(normalized):
            raise ValueError(
                f"Invalid file path: {filename}. "
                "Absolute paths are not allowed."
            )

        if normalized.startswith(".."):
            raise ValueError(
                f"Invalid file path: {filename}. "
                "Parent directory references (..) are not allowed."
            )

        # Step 3: Build full path and resolve it (handles case and symlinks)
        full_path = os.path.join(self.project_dir, normalized)

        # realpath() is KEY for Windows:
        # - Resolves symlinks
        # - Normalizes case (C:\ vs c:\)
        # - Resolves . and ..
        real_full_path = os.path.realpath(full_path)
        real_project_dir = os.path.realpath(self.project_dir)

        # Step 4: Ensure path is within project directory
        # Use os.path.commonpath to handle Windows case-insensitivity
        try:
            # commonpath raises ValueError if paths are on different drives
            common = os.path.commonpath([real_full_path, real_project_dir])

            # The common path must be exactly the project directory
            if os.path.normcase(common) != os.path.normcase(real_project_dir):
                raise ValueError(
                    f"File path escapes project directory: {filename}"
                )
        except ValueError:
            # Different drives on Windows (C:\ vs D:\)
            raise ValueError(
                f"File path is on a different drive: {filename}"
            )

        # Return the normalized path (not the real path, to preserve user's intent)
        return normalized

    def _execute_tool_call(self, tool_call):
        tool_name = tool_call.get("tool_name")
        args = tool_call.get("arguments", {})

        # Read tools
        if tool_name == "list_files":
            return list(self.files.keys())

        elif tool_name == "read_file":
            filename = self._validate_file_path(args.get("filename"))
            if filename not in self.files:
                return f"Error: file '{filename}' does not exist."
            return self.files[filename]

        # Write tools
        elif tool_name == "write_file":
            filename = self._validate_file_path(args.get("filename"))
            content = args.get("content")
            # ---  GUARDIAN LOGIC ---
            if filename in self.files:
                # The file exists. Is the agent making a mistake?
                # A simple heuristic: if the new content is much smaller than the old, it's suspicious.
                if len(content) < len(self.files[filename]) * 0.5:
                    return "Error: You are trying to use `write_file` on an existing file, which is destructive. This might be a mistake. Use `replace_text` or `insert_text` to make a modification. Action rejected."
            # ---  GUARDIAN LOGIC ---
            self.files[filename] = content
            return f"File '{filename}' was created/overwritten."

        elif tool_name == "append_to_file":
            filename = self._validate_file_path(args.get("filename"))
            content_to_append = args.get("content")
            if filename not in self.files:
                return f"Error: File '{filename}' does not exist to append content."
            self.files[filename] += "\n" + content_to_append
            return f"Content appended to '{filename}'."
        elif tool_name == "replace_lines":
            filename = args.get("filename")
            start_line = args.get("start_line") - 1  # Convert to 0-based index
            end_line = args.get("end_line") - 1
            new_lines_content = args.get("new_content")

            if filename not in self.files:
                return f"Error: File '{filename}' not found."

            lines = self.files[filename].splitlines()

            if start_line < 0 or end_line >= len(lines):
                return "Error: Line numbers are out of range."

            # Replace the specified lines
            lines[start_line: end_line + 1] = new_lines_content.splitlines()

            self.files[filename] = "\n".join(lines)
            return f"Lines {start_line+1} to {end_line+1} in '{filename}' were replaced."
        # Text manipulation tools (most important for refactoring)
        elif tool_name == "replace_text":
            filename = self._validate_file_path(args.get("filename"))
            old_text = args.get("old_text")
            new_text = args.get("new_text")
            # A surgical tool should only act on one thing at a time.
            count = args.get("count", 1)

            if filename not in self.files:
                return f"Error: File '{filename}' not found."

            original_content = self.files[filename]
            if old_text not in original_content:
                return f"Error: Text '{old_text[:50]}...' not found in '{filename}'."
            new_content = original_content.replace(old_text, new_text, count)
            if new_content == original_content:
                return f"Warning: No replacements made in '{filename}'. Text may not exist."

            self.files[filename] = new_content
            occurrences = original_content.count(old_text)
            return f"Text replaced in '{filename}' ({min(occurrences, count if count > 0 else occurrences)} occurrence(s))."

        elif tool_name == "delete_text_block":
            filename = self._validate_file_path(args.get("filename"))
            text_to_delete = args.get("text_to_delete")
            # This tool is essentially a replace with an empty string
            return self._execute_tool_call({
                "tool_name": "replace_text",
                "arguments": {"filename": filename, "old_text": text_to_delete, "new_text": ""}
            })
        elif tool_name == "autoformat_code":
            filename = self._validate_file_path(args.get("filename"))
            if filename not in self.files:
                return f"Error: File '{filename}' not found."

            # A simple implementation using a local formatter
            # A better one would use Docker for security and consistency
            try:
                # Assuming 'black' is installed in the orchestrator's environment
                result = subprocess.run(['black', os.path.join(
                    self.project_dir, filename)], capture_output=True, text=True)
                if result.returncode != 0:
                    return f"Error formatting file: {result.stderr}"

                # After formatting, we must re-read the file to update our in-memory state
                with open(os.path.join(self.project_dir, filename), 'r', encoding='utf-8') as f:
                    new_content = f.read()

                self.files[filename] = new_content

                return f"File '{filename}' was auto-formatted successfully."
            except Exception as e:
                return f"Failed to run autoformatter: {e}"
        elif tool_name == "create_directory":
            dirname = args.get("dirname")
            if not dirname:
                return "Error: folder name has not been specified."

            dirpath = os.path.join(self.project_dir, dirname)
            if os.path.exists(dirpath):
                return f"Folder '{dirname}' already exists."

            os.makedirs(dirpath)
            return f"Folder '{dirname}' successfully created."
        elif tool_name == "insert_text":
            filename = self._validate_file_path(args.get("filename"))
            content_to_insert = args.get("content_to_insert")
            # Optional: insert before this text
            before_text = args.get("before_text", None)
            # Optional: insert after this text
            after_text = args.get("after_text", None)

            if filename not in self.files:
                return f"Error: File '{filename}' not found."

            original_content = self.files[filename]

            if after_text and after_text in original_content:
                # Insert after the specified text
                parts = original_content.split(after_text, 1)
                new_content = parts[0] + after_text + \
                    "\n" + content_to_insert + parts[1]
                location_msg = f"after '{after_text[:30]}...'"
            elif before_text and before_text in original_content:
                # Insert before the specified text
                parts = original_content.split(before_text, 1)
                new_content = parts[0] + content_to_insert + \
                    "\n" + before_text + parts[1]
                location_msg = f"before '{before_text[:30]}...'"
            else:
                # Default: append to the end of the file
                new_content = original_content + "\n" + content_to_insert
                location_msg = "at end of file (anchor text not found)"

            self.files[filename] = new_content
            return f"Text inserted into '{filename}' {location_msg}."
        # return a high-level summary of a file's structure.
        elif tool_name == "read_structure":
            filename = self._validate_file_path(args.get("filename"))
            if not filename.endswith('.py'):
                return "Error: read_structure only supports Python files for now."
            if filename not in self.files:
                return f"Error: File '{filename}' not found."
            try:
                tree = ast.parse(self.files[filename])
                structure = []
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                        structure.append(f"Import: {ast.unparse(node)}")
                    elif isinstance(node, ast.FunctionDef):
                        structure.append(f"Function: {node.name}(...)")
                    elif isinstance(node, ast.ClassDef):
                        structure.append(f"Class: {node.name}")
                return "\n".join(structure)
            except Exception as e:
                return f"Error parsing file structure: {e}"
        else:
            return f"Error: Unknown Tool '{tool_name}'."

    def _build_index_from_disk(self):
        """Construit l'index RAG initial en lisant tous les fichiers suivis par Git."""
        self.logger.log("INFO", "Building initial code index from disk...")
        documents = []
        if not self.repo:
            self.code_index = VectorStoreIndex.from_documents([])
            return

        # Lister tous les fichiers suivis par Git
        tracked_files = self.repo.git.ls_files().split('\n')
        for filename in tracked_files:
            if not filename:
                continue
            filepath = os.path.join(self.project_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(text=content, metadata={"file_name": filename})
                documents.append(doc)
            except Exception:
                # Ignorer les fichiers qui ne peuvent pas être lus
                continue

        if not documents:
            self.code_index = VectorStoreIndex.from_documents([])
        else:
            self.code_index = VectorStoreIndex.from_documents(documents)
        self.logger.log("INFO", "Initial index built.")

    def _update_index_incrementally(self):
        """
        Met à jour l'index RAG de manière incrémentale en se basant sur le statut Git.
        Cette fonction doit être appelée APRÈS que l'agent a terminé ses modifications
        et AVANT le commit.
        """
        if not self.code_index:
            self._build_index_from_disk()
            return

        self.logger.log("INFO", "Updating RAG index incrementally...")

        # La manière de mettre à jour un index dépend de la bibliothèque (ex: LlamaIndex, ChromaDB)
        # L'idée générale est de supprimer les anciens documents et d'ajouter les nouveaux.
        # Pour LlamaIndex, la manière la plus simple reste de reconstruire, mais uniquement si nécessaire.
        # Une implémentation avancée utiliserait `index.delete_ref_doc()` et `index.insert()`.

        # Stratégie simplifiée mais efficace : reconstruire si des changements sont détectés.
        # C'est beaucoup mieux que de reconstruire à chaque appel d'outil.
        if self.repo.is_dirty(untracked_files=True):
            self.logger.log(
                "INFO", "Changes detected, rebuilding index from current disk state.")
            self._build_index_from_disk()
        else:
            self.logger.log(
                "INFO", "No changes detected, index is up-to-date.")

    def _rebuild_index_from_files(self):
        """Builds or rebuilds the index from the current state of self.files."""
        print("Building vector code index...")
        documents = []
        for filename, content in self.files.items():
            # Add the filename in metadata to retrieve it
            doc = Document(text=content, metadata={"file_name": filename})
            documents.append(doc)

        if not documents:
            # Create an empty index if no files exist
            self.code_index = VectorStoreIndex.from_documents([])
        else:
            self.code_index = VectorStoreIndex.from_documents(documents)
        print("Index built.")

    def _clean_workspace(self):
        """Wipes the workspace and initializes a fresh Git repository."""
        self.logger.log("INFO", f"Cleaning workspace: {self.project_dir}")
        if os.path.exists(self.project_dir):
            # shutil.rmtree can fail on Windows due to .git folder permissions
            def on_rm_error(func, path, exc_info):
                os.chmod(path, 0o777)
                func(path)
            shutil.rmtree(self.project_dir, onerror=on_rm_error)
        os.makedirs(self.project_dir)
        self.logger.log("INFO", "Initializing new Git repository...")
        self.repo = git.Repo.init(self.project_dir)

        # --- NEW: CREATE .gitignore AND INITIAL COMMIT ---
        gitignore_content = """
# Python
__pycache__/
*.pyc
.venv/
venv/
env/

# IDE files
.idea/
.vscode/

# Project Outputs
*.db
*.sqlite3

# Environment
.env
"""
        gitignore_path = os.path.join(self.project_dir, ".gitignore")
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)

        self.logger.log("INFO", "Creating initial commit with .gitignore...")
        self.repo.git.add(".gitignore")
        self.repo.index.commit("Initial commit: Add .gitignore")
        self.logger.log(
            "INFO", "Initial commit created. Git repository is ready.")

    def _save_state(self, last_completed_task_index):
        state = {
            "project_name": self.project_name,
            "plan": self.plan,
            "run_command": self.run_command,
            "last_completed_task_index": last_completed_task_index,
            "user_prompt": self.user_prompt,
            "cost_tracker_state": self.cost_tracker.to_dict(),
            "metrics_tracker_state": self.metrics_tracker.to_dict()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"State saved. Task {last_completed_task_index + 1} finished.")

    def _load_state(self):
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        self.project_name = state["project_name"]
        self.plan = state["plan"]
        self.run_command = state["run_command"]
        self.last_completed_task_index = state["last_completed_task_index"]
        self.user_prompt = state["user_prompt"]

        cost_tracker_state = state.get("cost_tracker_state")
        if cost_tracker_state:
            self.cost_tracker.from_dict(cost_tracker_state)
        metrics_state = state.get("metrics_tracker_state")
        if metrics_state:
            self.metrics_tracker.from_dict(metrics_state)
        self._build_index_from_disk()

    def _get_project_structure_string(self):
        """
        Generates a visually appealing and accurate string representation
        of the project's file and directory structure.
        """
        # Step 1: Build a nested dictionary representing the file structure
        file_tree = {}
        for filepath in sorted(self.files.keys()):
            parts = filepath.split('/')
            current_level = file_tree
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

        # Step 2: Recursively format the dictionary into a tree string
        def format_tree(d, prefix=""):
            lines = []
            entries = sorted(d.keys())
            for i, entry in enumerate(entries):
                connector = "└── " if i == len(entries) - 1 else "├── "
                lines.append(prefix + connector + entry)
                if d[entry]:  # If it's a directory (has children), recurse
                    extension = "    " if i == len(entries) - 1 else "│   "
                    lines.extend(format_tree(d[entry], prefix + extension))
            return lines

        # Start the process
        tree_lines = format_tree(file_tree)
        return ".\n" + "\n".join(tree_lines)

    def _run_clarification_phase(self):
        self.logger.log_phase("Requirements Clarification")
        # We'll treat the whole clarification process as one task.
        self.logger.log_task(1, 1, "Clarify and validate user requirements")
        self.logger.start_step(1, 1)

        clarification_history = ""
        current_prompt = self.user_prompt

        for i in range(5):  # Limit to 5 rounds of questions to prevent infinite loops
            analysis = self.requirements_agent.analyze_requirements(
                current_prompt, clarification_history)
            self.cost_tracker.calculate_and_print_cost(self.model_name)

            questions = analysis.get("questions", [])
            refined_prompt = analysis.get("refined_prompt", current_prompt)
            # Log the agent's output for this step
            self.logger.log_event_in_step("thought", {
                                          "thought": f"Analyzed requirements. Refined prompt and found {len(questions)} questions."})
            self.logger.log_event_in_step("clarification_output", {
                "questions": questions,
                "refined_prompt": refined_prompt
            })
            if not questions:
                self.logger.log(
                    "INFO", "Requirements are clear. Clarification phase complete.")
                self.user_prompt = refined_prompt  # Set the final validated prompt
                break

            print("\n🤔 The AI has some clarifying questions for you:")
            for q in questions:
                print(f"- {q}")

            print("\n(You can answer the questions, provide more details, or type 'ok' if you're satisfied with the refined prompt)")
            user_response = input("> ")
            # Log the user's interaction
            self.logger.log_event_in_step(
                "user_interaction", {"response": user_response})

            if user_response.lower().strip() in ["ok", "y", "yes", "proceed"]:
                self.logger.log(
                    "INFO", "User approved requirements. Proceeding.")
                self.user_prompt = refined_prompt
                break

            clarification_history += f"\nPrevious Refined Prompt:\n{refined_prompt}\n"

            joined_questions = '\n- '.join(questions)
            clarification_history += f"AI Questions:\n- {joined_questions}\n"
            clarification_history += f"User Answers:\n{user_response}\n"

            # The user's new response becomes the prompt for the next analysis round
            current_prompt = user_response
        else:

            self.logger.log(
                "WARNING", "Max clarification rounds reached. Proceeding with the latest requirements.")
            self.user_prompt = refined_prompt
        # Save the final, validated requirements to the file
        with open(self.requirements_file, "w", encoding="utf-8") as f:
            f.write(
                f"# Final Validated Requirements\n\n{self.user_prompt}\n\n---\n\n")
            if clarification_history != "":
                f.write(f"## Conversation History\n\n{clarification_history}")
        self.logger.log(
            "INFO", f"Final requirements saved to {self.requirements_file}")

    async def run(self):
        project_completed_successfully = False
        try:
            # --- PHASE 0: REQUIREMENTS CLARIFICATION ---
            if not self.plan:  # Only run this if we are starting a new project/plan
                self._run_clarification_phase()
            if not self.plan:
                self.logger.log_phase("Architectural Design")
                self.logger.log_task(
                    1, 1, "Designing the software architecture")
                self.logger.start_step(1, 1)

                self.technical_architecture = self.architect_agent.design_architecture(
                    self.user_prompt, self.logger)
                self.cost_tracker.calculate_and_print_cost(self.model_name)

                if not self.technical_architecture:
                    self.logger.log(
                        "ERROR", "Architect Agent failed to design an architecture. Halting.")
                    return

                self.logger.log(
                    "INFO", f"Architecture designed: {self.technical_architecture}")

            if not self.plan:
                self.logger.log_task(
                    1, 1, "Generate project plan from user requirements using SADT/SART")
                # Treat planning as a single-step task
                self.logger.start_step(1, 1)
                print("--- 🤖 Phase 1: SADT/SART Planning ---")
                self.plan, self.run_command = self.sadt_sart_agent.create_plan(
                    self.user_prompt, self.technical_architecture, self.logger)

                self.cost_tracker.calculate_and_print_cost(self.model_name)

                print("result", self.plan, self.run_command)

                if not self.plan or not self.run_command:
                    self.logger.log(
                        "ERROR", "The Product Owner failed to generate a plan. Halting.")
                    return
                self.last_completed_task_index = -1  # No task is done yet
                self.logger.log(
                    "INFO", f"Plan generated with {len(self.plan)} tasks.")
                for n in range(len(self.plan)):
                    print(f"{n+1} {self.plan[n]}")
                print(f"🚀 Launch command: {self.run_command}")
                # Once the plan is created, initialize an empty index
                self._rebuild_index_from_files()

            #    self.conversation_history.append(
            #        {"role": "PO", "content": f"Plan: {self.plan}"})

            # --- Phase 2: Development & Per-Task Review Cycle ---
            self.logger.log_phase("Development")
            start_task_index = self.last_completed_task_index + 1
            total_tasks = len(self.plan)  # total_tasks variable defined here

            # for i, task in enumerate(self.plan[start_task_index:], start=start_task_index):
            current_task_index = self.last_completed_task_index + 1

            with open(self.requirements_file, 'r', encoding='utf-8') as f:
                requirements_content = f.read()

            technical_architecture_json = json.dumps(
                self.technical_architecture, indent=2)

            while current_task_index < len(self.plan):

                task = self.plan[current_task_index]
                self.metrics_tracker.start_task(task)
                self.logger.log_task(current_task_index, total_tasks, task)
                self.logger.start_step(1, 1)
                rag_context = self._get_rag_context(task)

                completed_tasks_list = self.plan[:current_task_index]
                upcoming_tasks_list = self.plan[current_task_index + 1:]
                project_plan_context = "Tasks Completed:\n"
                for t in completed_tasks_list:
                    project_plan_context += f"- [x] {t}\n"
                if len(completed_tasks_list) == 0:
                    project_plan_context += "None"

                project_plan_context += f"\n**>> CURRENT TASK: [ ] {task} <<**\n"

                project_plan_context += "\nUpcoming Tasks:\n"
                for t in upcoming_tasks_list:
                    project_plan_context += f"- [ ] {t}\n"

                full_prompt_for_dev = f"""
**--- PROJECT BRIEFING ---**

**1. Mission Statement (The 'Why'):**
{requirements_content}

**2. Technical Blueprint (The 'How'):**
```json
{technical_architecture_json}
```

3. Project Plan (The 'What'):
{project_plan_context}

** YOUR CURRENT TASK ASSIGNMENT**

Task Description: {task}

Relevant Code Context (from RAG):
{self._get_rag_context(task)}
"""
                # Pass a boolean to tell the tester if this is the final test
                task_development_success, agent_final_response, tool_calls = await self.dev_agent.execute_task(full_prompt_for_dev)
                for tool_call in tool_calls:
                    self.metrics_tracker.track_tool_call(
                        tool_call['tool_name'], True)  # FIXME: always return success
                # We need a better way to check for success, but for now, we assume
                # if it doesn't crash, it succeeded in its own view.
                if not task_development_success:
                    self.logger.log(
                        "ERROR", f"Agent failed to complete the development for task '{task}'. Halting.")
                    return

                # --- IMMEDIATE CODE REVIEW LOOP FOR THIS TASK ---
                max_review_cycles_per_task = 3
                review_passed = False
                for cycle in range(max_review_cycles_per_task):
                    self.logger.log(
                        "INFO", f"Starting review cycle {cycle + 1}/{max_review_cycles_per_task} for task '{task}'")
                    self.metrics_tracker.increment_review_cycle(task)

                    git_diff = self._get_project_diff_since_last_commit()  # New method needed!
                    if not git_diff.strip():
                        self.logger.log(
                            "INFO", "No file changes detected for this task. Review skipped.")
                        review_passed = True
                        break
                    # Treat each review cycle as a step
                    self.logger.start_step(
                        cycle + 1, max_review_cycles_per_task)
                    self.logger.log_event_in_step(
                        "git_diff", {"diff": git_diff})

                    # Pass the previous issues to the reviewer so it has memory!
                    previous_issues = issues if cycle > 0 else None
                    issues, reviewer_tool_usage = await self.reviewer_agent.review_code(task,
                                                                                        self._get_project_structure_string(), git_diff,  previous_issues
                                                                                        )

                    if reviewer_tool_usage:
                        self.logger.log(
                            "INFO", f"CodeReviewerAgent used {len(reviewer_tool_usage)} tool(s) during investigation.")
                        for tool_call in reviewer_tool_usage:
                            tool_name = tool_call.get(
                                'tool_name', 'unknown_tool')
                            tool_result = tool_call.get('result', '')

                            # Log the detailed event for traceability
                            self.logger.log_event_in_step(
                                "tool_result",
                                {
                                    "agent": "CodeReviewerAgent",
                                    "tool_name": tool_name,
                                    "arguments": tool_call.get('arguments', {}),
                                    "result": tool_result
                                }
                            )
                            # Always a success for the reviewer since they are read-only
                            self.metrics_tracker.track_tool_call(
                                tool_name, success=True)
                    self.cost_tracker.calculate_and_print_cost(self.model_name)
                    self.logger.log(
                        "INFO", f"Reviewer found {len(issues)} issues.")  # FIXME

                    self.logger.log_event_in_step(
                        "review_result", {"issues": issues})  # FIXME

                    if not issues:
                        self.logger.log(
                            "INFO", "Code review for this task passed!")
                        # FIXME code dbg
                        print(f" review_passed_on_first_try={cycle == 0}")
                        self.metrics_tracker.complete_task(
                            task, success=True, review_passed_on_first_try=(cycle == 0))
                        review_passed = True
                        break

                    # !!!! IMPORTANT FIX ISSUES ONE BY ONE !!!!
                    self.logger.log(
                        "WARNING", f"Review found {len(issues)} issues. Attempting to fix them one by one.")

                    surgical_fixes_issues = [iss for iss in issues if iss.get(
                        'type') in ['typo', 'style_violation', 'import_error', 'naming_convention']]
                    complex_dev_issues = [iss for iss in issues if iss.get(
                        'type') not in ['typo', 'style_violation', 'import_error', 'naming_convention']]
                    if complex_dev_issues:
                        # if a complex problem persists, we don't spend time with minor corrections
                        # We instantly launch replan.
                        self.logger.log(
                            "WARNING", f"Found {len(complex_dev_issues)} complex issues. The plan is invalid. Invoking Product Owner for replanning.")
                        for issue in complex_dev_issues:
                            print(issue)
                        # Récupérer les tâches déjà complétées pour donner le contexte au PO
                        completed_tasks = self.plan[:current_task_index]

                        # Appeler le PO pour qu'il crée un nouveau plan
                        new_plan_tasks = self.po_agent.replan_project(
                            self.user_prompt,  # Exigences fonctionnelles
                            self.plan,        # Le plan original complet
                            completed_tasks,  # Ce qui a déjà été fait
                            task,             # La tâche qui a échoué
                            complex_dev_issues,  # Pourquoi elle a échoué
                            self.logger
                        )

                        if new_plan_tasks is None:
                            self.logger.log(
                                "ERROR", "Replanning failed. Halting execution.")
                            sys.exit(1)

                        self.logger.log(
                            "INFO", "Product Owner has created a new plan:")
                        for i, new_task_item in enumerate(new_plan_tasks):
                            print(f"  {i+1}. {new_task_item}")

                        # Remplacer les anciennes tâches restantes par le nouveau plan
                        self.plan = completed_tasks + new_plan_tasks

                        # FIXME debug
                        for i, new_task_item in enumerate(self.plan):
                            print(f"  {i+1}. {new_task_item}")

                        # La tâche actuelle a échoué et a été remplacée. La boucle continuera
                        # à partir de `current_task_index`, qui est maintenant la première nouvelle tâche du plan corrigé.
                        review_passed = False  # S'assurer que la tâche actuelle est marquée comme échouée
                        self.logger.log(
                            "ERROR", f"Task '{task}' failed and was replanned. Continuing with the new plan.")
                        # Sortir de la boucle de revue
                        break
                    elif surgical_fixes_issues:
                        all_surgical_fixes_succeeded = True
                        self.logger.log(
                            "INFO", f"No complex issues found. Attempting to fix {len(surgical_fixes_issues)} simple issues...")

                        fix_task = f"Your previous work for '{task}' had issues: {json.dumps(surgical_fixes_issues)}. Please fix them."
                        fix_success, _, tool_calls = await self.dev_agent.execute_task(fix_task)
                        if not fix_success:
                            self.logger.log(
                                "ERROR", f"Agent failed to fix issues: {issues}")
                            all_surgical_fixes_succeeded = False

                        if not all_surgical_fixes_succeeded:
                            break  # Exit the review loop, the task failed
                    else:
                        # No problem found
                        review_passed = True
                        break
                # --- DECISION POINT: COMMIT OR HALT ---
                if review_passed:
                    # BEFORE  commit, update RAG index with changes.
                    self._update_index_incrementally()
                    commit_message = f"feat: Complete and review task '{task}'"
                    self._git_commit(commit_message)
                    self.metrics_tracker.complete_task(
                        task, success=True, review_passed_on_first_try=(cycle == 0))
                    self._save_state(current_task_index)
                    self.logger.log(
                        "INFO", "Task approved, committed, and state saved.")
                    current_task_index += 1

                else:
                    # Task failed review - do NOT commit
                    self.logger.log(
                        "ERROR", f"Task '{task}' failed code review after {cycle+1} attempt(s).")

                    if complex_dev_issues:
                        # New tasks were added to plan - continue to fix them
                        self.logger.log(
                            "INFO", "Complex issues found. New corrective tasks added to plan. Continuing...")
                        self.metrics_tracker.complete_task(
                            task, success=False, review_passed_on_first_try=False)
                    else:
                        # Simple issues couldn't be fixed - halt execution
                        self.logger.log(
                            "ERROR", "Failed to fix issues. Halting execution.")
                        self.metrics_tracker.complete_task(
                            task, success=False, review_passed_on_first_try=False)
                        sys.exit(1)

                    # sys.exit(1)
                    # return

            project_completed_successfully = True

        finally:
            self.logger.log_phase("Finalization")

            # --- Final Validation ---
            if self.tester_agent:
                self.logger.log_phase("Final Validation")
                success, output = self.tester_agent.run_quality_gate(
                    self.technical_architecture, is_last_task=True)
                if not success:
                    print(f"❌ Final validation failed: {output}")
                    project_completed_successfully = False
                else:
                    print("✅ Final validation passed!")
            if project_completed_successfully:
                # --- NEW UNIT TESTING PHASE (Optional) ---
                if self.run_tests_enabled:
                    self.logger.log_phase("Unit Test Generation")
                    print("\n--- 🔬 Phase 3: Unit Test Generation ---")
                    # We'll generate tests for all .py files except requirements
                    files_to_test = {
                        k: v for k, v in self.files.items() if k.endswith(".py")}

                    for filename, content in files_to_test.items():
                        print(f"Generating tests for {filename}...")
                        try:
                            test_content = self.unit_test_agent.write_tests(
                                filename, content, self.logger)
                            self.cost_tracker.calculate_and_print_cost(
                                self.model_name)

                            test_filename = f"test_{filename}"
                            self._execute_tool_call({
                                "tool_name": "write_file",
                                "arguments": {"filename": test_filename, "content": test_content}
                            })
                        except Exception as e:
                            print(
                                f"❌ Failed to generate tests for {filename}: {e}")

                # --- VALIDATION PHASE (Now includes running tests) ---
                if self.tester_agent is not None:
                    print("\n--- ⚙️ Final Validation Phase ---")
                    success, output = self.tester_agent.execute(
                        self.run_command, is_last_task=True)
                    if not success:
                        print(f"❌ Final validation failed: {output}")
                        # Mark as incomplete if final tests fail
                        project_completed_successfully = False
            # --- NEW DOCUMENTATION PHASE ---
            if project_completed_successfully:
                self.logger.log_phase("Documentation Generation")
                self.logger.log_task(1, 1, "Writing final project README.md")
                self.logger.start_step(1, 1)
                try:
                    project_structure = self._get_project_structure_string()
                    requirements_txt = "# No requirements.txt found"
                    with open(os.path.join(self.project_path, "requirements.txt"), 'r', encoding='utf-8') as f:
                        requirements_txt = f.read()

                    readme_content = self.doc_agent.write_documentation(
                        self.project_name,
                        project_structure,
                        self.run_command,
                        requirements_txt, self.logger
                    )

                    print("🧠 Agent thought: Generated README.md content.")

                    # Use the 'write_file' tool to save the documentation
                    self._execute_tool_call({
                        "tool_name": "write_file",
                        "arguments": {
                            "filename": "README.md",
                            "content": readme_content
                        }
                    })
                    print("✅ Documentation (README.md) created successfully.")

                except Exception as e:
                    print(
                        f"❌ An error occurred during documentation generation: {e}")
            # This block will ALWAYS execute, whether the script succeeds or fails,
            # which guarantees you'll always have your cost report.
            cost_summary = self.cost_tracker.get_summary()
            self.logger.log("INFO", cost_summary)
            metrics_summary = self.metrics_tracker.get_summary()
            self.logger.log("INFO", metrics_summary)
            self.logger.log_final_summary(cost_summary, metrics_summary)
            self.logger.write_to_file()

    def _get_rag_context(self, query):
        if self.code_index:
            retriever = self.code_index.as_retriever(similarity_top_k=3)
            relevant_nodes = retriever.retrieve(query)

            if not relevant_nodes:
                return "No relevant code snippets found in the project for this task."

            context_str = "Here are relevant code snippets for the current task:\n\n"
            for node in relevant_nodes:
                # La métadonnée 'file_name' est cruciale
                filename = node.metadata.get('file_name', 'unknown_file')
                context_str += f"--- Snippet from {filename} ---\n"
                context_str += node.get_content() + "\n-----------------------------------\n"
            return context_str
        return "No code exists yet."


# Script entry point
if __name__ == "__main__":
    import asyncio  # Add this import at the top if not already there

    # Usage: python orchestrator.py <project_name> [--new] [--google] ...
    # Ex: python orchestrator.py contract_manager_app --new
    # Ex: python orchestrator.py new_website --prompt initial_ideas.txt

    project_name = sys.argv[1]
    initial_prompt_file = None
    if "--prompt" in sys.argv:
        try:
            initial_prompt_file = sys.argv[sys.argv.index("--prompt") + 1]
        except IndexError:
            print("Error: --prompt flag requires a filename.")
            sys.exit(1)

    provider = "ollama"  # default
    if "--google" in sys.argv:
        provider = "google"
    # We could add "--openai", in the future

    force_new = "--new" in sys.argv
    run_tests = "--with-tests" in sys.argv
    orchestrator = Orchestrator(project_name, initial_prompt_file,
                                provider=provider, force_new=force_new, run_tests=run_tests)
    asyncio.run(orchestrator.run())
