import json
import os
import subprocess
import sys
from typing import Dict, Any, List, Literal
from tools import get_code_summary
from tools.universal_refactoring import add_code_block as _add_code_block,    refactor_rename_symbol as _refactor_rename_symbol, delete_code_block as _delete_code_block

from pydantic import BaseModel, Field


class ReviewIssue(BaseModel):
    """Single code review issue."""
    severity: Literal["critical", "major", "minor", "suggestion"] = Field(
        description="Severity level of the issue"
    )
    type: Literal[
        "typo", "style_violation", "import_error", "naming_convention",
        "missing_feature", "incomplete_feature", "logic_bug", "integration_issue"
    ] = Field(
        description="Category of the issue"
    )
    file: str = Field(
        description="Full path of the file containing the issue"
    )
    line: int = Field(
        description="Line number where the issue occurs (approximate)"
    )
    description: str = Field(
        description="Clear one-sentence explanation of what is wrong"
    )
    suggestion: str = Field(
        description="Simple recommendation on how to fix the issue"
    )


class ReviewReport(BaseModel):
    """The complete set of issues found in a code review. This is the required input for the finish_review tool."""
    issues: List[ReviewIssue] = Field(
        description="A list of all issue objects found. If no issues were found, this MUST be an empty list []."
    )


class OrchestratorTools:
    def __init__(self, orchestrator_instance):
        """
        Initializes the tool suite with a reference to the main orchestrator.
        This allows tools to access the orchestrator's state (files, project_dir, etc.).
        """
        self.orchestrator = orchestrator_instance

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
        full_path = os.path.join(self.orchestrator.project_dir, normalized)

        # realpath() is KEY for Windows:
        # - Resolves symlinks
        # - Normalizes case (C:\ vs c:\)
        # - Resolves . and ..
        real_full_path = os.path.realpath(full_path)
        real_project_dir = os.path.realpath(self.orchestrator.project_dir)

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

    def add_code_block(
        self,
        filepath: str,
        new_code: str,
        location: str = "end_of_file",
        target_name: str = None
    ) -> dict:
        """Add code block at semantic location."""
        filename = self._validate_file_path(filepath)
        filepath = os.path.join(self.orchestrator.project_dir, filename)
        if not os.path.exists(filepath):
            return {
                "success": False,
                "error": f"File {filename} does not exist, use tool write_file instead",
            }
        return _add_code_block(filepath, new_code, location, target_name)

    def refactor_rename_symbol(
        self,
        filepath: str,
        old_name: str,
        new_name: str,
        symbol_type: str = "all"
    ) -> dict:
        """Safely rename a symbol."""
        filepath = self._validate_file_path(filepath)
        filepath = os.path.join(self.orchestrator.project_dir, filepath)
        return _refactor_rename_symbol(filepath, old_name, new_name, symbol_type)

    def delete_code_block(
        self,
        filepath: str,
        block_name: str,
        block_type: str = "auto"
    ) -> dict:
        """Delete a code block by name."""
        filepath = self._validate_file_path(filepath)
        filepath = os.path.join(self.orchestrator.project_dir, filepath)

        return _delete_code_block(filepath, block_name, block_type)

    def replace_text(self, filename: str, old_text: str, new_text: str, count: int = 1) -> str:
        filename = self._validate_file_path(filename)

        # A surgical tool should only act on one thing at a time.
        # count = 1
        original_content = self.read_file(filename)
        if old_text not in original_content:
            return {
                "success": False,
                "error": f"Error: Text '{old_text[:50]}...' not found in '{filename}'."
            }
        new_content = original_content.replace(old_text, new_text, count)
        if new_content == original_content:
            return {
                "success": False,
                "error": f"Warning: No replacements made in '{filename}'. Text may not exist."
            }
        self.write_file(filename, new_content)
        occurrences = original_content.count(old_text)
        return {
            "success": True,
            "message": f"Text replaced in '{filename}' ({min(occurrences, count if count > 0 else occurrences)} occurrence(s))."
        }

    def insert_text(self, filename: str, content_to_insert: str, before_text: str = None, after_text: str = None) -> str:
        """
        Inserts a block of text into a file, either relative to anchor text or at the beginning/end.

        Args:
            filename: The path to the file.
            content_to_insert: The text to be inserted.
            before_text: Anchor text. The content will be inserted immediately before the first occurrence of this text.
                        If set to "", the content is inserted at the absolute beginning of the file.
            after_text: Anchor text. The content will be inserted immediately after the first occurrence of this text.
                        If set to "", the content is inserted at the absolute end of the file.

        Returns:
            A string describing the result of the operation.
        """
        filename = self._validate_file_path(filename)
        original_content = self.read_file(filename)

        # Gérer les cas où le fichier n'existe pas ou est vide
        if "Error: file" in original_content:
            return original_content  # Retourne le message d'erreur de read_file

        # FIX 1: Gérer les ancres vides comme des positions absolues (début/fin)
        if before_text == "":
            new_content = content_to_insert + "\n" + original_content
            location_msg = "at the beginning of the file"

        elif after_text == "":
            new_content = original_content + "\n" + content_to_insert
            location_msg = "at the end of the file"

        # FIX 2: La logique existante pour les ancres non vides est conservée, mais vient après
        elif after_text is not None and after_text in original_content:
            # Insert after the specified text
            parts = original_content.split(after_text, 1)
            new_content = parts[0] + after_text + \
                "\n" + content_to_insert + parts[1]
            location_msg = f"after '{after_text[:30]}...'"

        elif before_text is not None and before_text in original_content:
            # Insert before the specified text
            parts = original_content.split(before_text, 1)
            new_content = parts[0] + content_to_insert + \
                "\n" + before_text + parts[1]
            location_msg = f"before '{before_text[:30]}...'"

        else:
            # FIX 3: La logique par défaut est maintenant plus intelligente.
            # Si une ancre était fournie mais non trouvée, le message d'erreur est plus clair.
            if before_text is not None or after_text is not None:
                # On ajoute quand même à la fin, mais on prévient que l'ancre n'a pas été trouvée
                new_content = original_content + "\n" + content_to_insert
                location_msg = "at end of file (anchor text not found)"
            else:
                # Comportement par défaut si aucune ancre n'est spécifiée : ajouter à la fin
                new_content = original_content + "\n" + content_to_insert
                location_msg = "at end of file (no anchor specified)"

        self.write_file(filename, new_content)
        return {"success": True, "message": f"Text inserted into '{filename}' {location_msg}."}

    def read_file(self, filename: str) -> dict:
        filename = self._validate_file_path(filename)
        filepath = os.path.join(self.orchestrator.project_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return {"success": False, "error": f"Error: file '{filename}' does not exist."}
        except Exception as e:
            return {"success": False, "error": f"Error reading file '{filename}': {e}"}

    def create_directory(self, dirname: str) -> dict:
        if not dirname:
            return {"success": False, "error": "folder name has not been specified."}
        dirpath = os.path.join(self.orchestrator.project_dir, dirname)
        if os.path.exists(dirpath):
            return {"success": False, "error": f"Folder '{dirname}' already exists."}
        os.makedirs(dirpath)
        return {"success": True, "message": f"Folder '{dirname}' successsfully created."}

    def list_files(self):
        # Lit la liste des fichiers directement depuis le disque (gérés par Git)
        if not self.orchestrator.repo:
            return "Error: Git repository not initialized."
        return self.orchestrator.repo.git.ls_files().split('\n')

    def autoformat_code(self, filename) -> dict:
        filename = self._validate_file_path(filename)
        filepath = os.path.join(self.orchestrator.project_dir, filename)
        # A simple implementation using a local formatter
        # A better one would use Docker for security and consistency
        try:
            if filename.endswith(".py"):
                # Assuming 'black' is installed in the orchestrator's environment
                result = subprocess.run(
                    ['black', filepath], capture_output=True, text=True)
                if result.returncode != 0:
                    return {"success": False, "error": f"Error formatting file: {result.stderr}"}
                return {"success": True, "message": f"File '{filename}' was auto-formatted successfully."}
            elif filename.endswith((".ts", ".tsx", ".js", ".jsx")):
                # Assuming 'prettier' is installed in the orchestrator's environment
                result = subprocess.run(
                    ['prettier', '--write', filepath], capture_output=True, text=True)
                if result.returncode != 0:
                    return {"success": False, "error": f"Error formatting file: {result.stderr}"}
                return {"success": True, "message": f"File '{filename}' was auto-formatted successfully."}
            elif filename.endswith((".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")):
                # Assuming 'clang-format' is installed in the orchestrator's environment
                result = subprocess.run(
                    ['clang-format', '-i', filepath], capture_output=True, text=True)
                if result.returncode != 0:
                    return {"success": False, "error": f"Error formatting file: {result.stderr}"}
                return {"success": True, "message": f"File '{filename}' was auto-formatted successfully."}
            else:
                return {"success": False, "error": f"no autoformatter provider for {filename}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to run autoformatter: {e}"}

    def delete_file(self, filename: str) -> dict:
        """
        Securely deletes a file within the project workspace.

        Args:
            filename: The path of the file to delete, relative to the project root.

        Returns:
            A dictionary indicating the success or failure of the operation.
        """
        try:
            # SECURITY: Validate the path is within the project and contains no ".." traversal.
            validated_filename = self._validate_file_path(filename)
            filepath = os.path.join(self.project_dir, validated_filename)

            # Check 1: Does the path exist?
            if not os.path.exists(filepath):
                return {"success": False, "error": f"File '{validated_filename}' does not exist and cannot be deleted."}

            # Check 2: Is it a file and not a directory?
            if not os.path.isfile(filepath):
                return {"success": False, "error": f"Path '{validated_filename}' is a directory, not a file. Use a 'delete_directory' tool instead."}

            # Delete the file
            os.remove(filepath)

            # CRITICAL UPDATE: The RAG index must be updated to "forget" this file.
            # The simplest way is to rebuild the index from the current disk state.
            self.logger.log(
                "INFO", f"File '{validated_filename}' deleted. Updating RAG index.")
            self._build_index_from_disk()

            return {"success": True, "message": f"File '{validated_filename}' was successfully deleted."}

        except ValueError as e:
            # Error raised by _validate_file_path
            return {"success": False, "error": str(e)}
        except Exception as e:
            # Other potential errors (e.g., file system permissions)
            return {"success": False, "error": f"An unexpected error occurred while deleting file '{filename}': {e}"}

    def write_file(self, filename: str, content: str) -> str:
        filename = self._validate_file_path(filename)
        # ---  GUARDIAN LOGIC ---
        filepath = os.path.join(self.orchestrator.project_dir, filename)
        try:
            # S'assurer que le répertoire existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                # The file exists. Is the agent making a mistake?
                # A simple heuristic: if the new content is much smaller than the old, it's suspicious.
                if len(content) < len(existing_content) * 0.5:
                    print("Error: You are trying to use `write_file` on an existing file, which is destructive. This might be a mistake. Use `replace_text` or `insert_text` to make a modification. Action rejected.")
                    return {"success": False, "error": "You are trying to use `write_file` on an existing file, which is destructive. This might be a mistake. Use `replace_text` or `insert_text` to make a modification. Action rejected."}
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            result = self.autoformat_code(filename)
            print(result)  # FIXME: to remove
            # Pas besoin de mettre à jour le RAG ici, on le fera après la tâche
            return {"success": True, "message": f"File '{filename}' was created/overwritten on disk."}
        except Exception as e:
            return {"success": False, "message": f"Error writing file '{filename}': {e}"}

    def get_code_summary(self, filename: str) -> Dict[str, Any]:
        print("required filename", filename)
        filename = self._validate_file_path(filename)
        filepath = os.path.join(self.orchestrator.project_dir, filename)
        summary = get_code_summary(filepath)
        return json.dumps(summary, indent=2, ensure_ascii=False)

    def submit_review(self, report: ReviewReport) -> dict:
        """
         Call this tool ONLY when your investigation is complete to submit your final report.

         Args:
             report: A ReviewReport object containing a list of all issues found.
         """
        # The 'report' argument will be a validated Pydantic object
        print("[DEBUG] submit_review input")
        print(report)
        if not isinstance(report, ReviewReport):
            validated_report = ReviewReport.model_validate(report)
        else:
            validated_report = report
        print("validated input")
        print(validated_report)
        issues_list = [issue.model_dump() for issue in validated_report.issues]
        print("[DEBUG] submit_review output ", issues_list)
        return {
            "success": True,
            "message": f"Review submitted with {len(issues_list)} issue(s)",
            "issues": issues_list
        }

    async def call_agent(self, agent_name: str, message: str) -> str:
        """
        Call another agent to ask for help, clarification, or a review.
        Args:
            agent_name: The name of the agent to call (e.g., "CodeReviewerAgent", "ProductOwnerAgent", "SoftwareArchitectAgent").
            message: The question or request for the agent.
        """
        agent = self.orchestrator.get_agent_by_name(agent_name)
        if not agent:
            return f"Error: Agent '{agent_name}' not found. Available agents: {self.orchestrator.get_available_agent_names()}"

        # We use the standardized chat method
        try:
            response = await agent.chat(message)

            return response
        except Exception as e:
            return f"Error calling agent {agent_name}: {str(e)}"
