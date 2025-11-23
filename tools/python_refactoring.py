"""
Semantic refactoring tools for Python code.
Uses AST (Abstract Syntax Tree) for safe, intelligent code modifications.
"""

import ast
import os
import sys
from typing import Dict, Any, Optional, List, Literal
from pathlib import Path


class PythonRefactoring:
    """Safe refactoring operations for Python code using AST."""

    def __init__(self, filepath: str):
        """
        Initialize refactoring tool for a Python file.

        Args:
            filepath: Path to the Python file to refactor
        """
        self.filepath = filepath

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            self.source_code = f.read()

        try:
            self.tree = ast.parse(self.source_code)
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax in {filepath}: {e}")

        self.lines = self.source_code.split('\n')
        self.indent_style = self._detect_indentation()
        self.python_version = sys.version_info

    def _detect_indentation(self) -> str:
        """
        Detect the indentation style used in the file.

        Returns:
            The indentation string (spaces or tab) used in the file
        """
        # Look for the first indented line to detect style
        indent_samples = []

        for line in self.lines:
            if line and line[0] in ' \t':
                # Extract leading whitespace
                indent = ''
                for char in line:
                    if char in ' \t':
                        indent += char
                    else:
                        break

                if indent:
                    indent_samples.append(indent)

        if not indent_samples:
            return '    '  # Default to 4 spaces

        # Check if tabs are used
        if any('\t' in indent for indent in indent_samples):
            return '\t'

        # Find the most common indentation level (GCD of all indents)
        from math import gcd
        from functools import reduce

        indent_lengths = [len(indent)
                          for indent in indent_samples if len(indent) > 0]

        if not indent_lengths:
            return '    '

        # Find GCD of all indentation lengths to determine base unit
        common_indent = reduce(gcd, indent_lengths)

        # Prefer common values (2, 4, 8)
        if common_indent in [2, 4, 8]:
            return ' ' * common_indent
        elif common_indent == 1:
            # If GCD is 1, look for most common value
            from collections import Counter
            most_common = Counter(indent_lengths).most_common(1)[0][0]
            return ' ' * most_common
        else:
            return ' ' * common_indent

    def _get_indentation_level(self, line: str) -> str:
        """
        Get the indentation of a specific line.

        Args:
            line: The line to analyze

        Returns:
            The indentation string for that line
        """
        indent = ''
        for char in line:
            if char in ' \t':
                indent += char
            else:
                break
        return indent

    def _indent_code(self, code: str, base_indent: str) -> str:
        """
        Properly indent multi-line code, preserving relative indentation.

        Args:
            code: The code to indent
            base_indent: The base indentation to apply

        Returns:
            Indented code
        """
        lines = code.rstrip().split('\n')
        if not lines:
            return ''

        # Find the minimum indentation in the code (excluding blank lines)
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, current_indent)

        if min_indent == float('inf'):
            min_indent = 0

        # Apply indentation while preserving relative spacing
        result_lines = []
        for line in lines:
            if not line.strip():
                # Preserve blank lines without trailing spaces
                result_lines.append('')
            else:
                # Remove base indentation and add new base indentation
                current_indent = len(line) - len(line.lstrip())
                relative_indent = current_indent - min_indent
                new_line = base_indent + \
                    (' ' * relative_indent) + line.lstrip()
                result_lines.append(new_line)

        return '\n'.join(result_lines)

    def _find_node_by_name(self, name: str, node_type: Optional[str] = None) -> List[ast.AST]:
        """
        Find all nodes with a given name (supports nested classes).

        Args:
            name: Name to search for
            node_type: Optional filter by node type ("class", "function", or None for any)

        Returns:
            List of matching nodes
        """
        matches = []

        for node in ast.walk(self.tree):
            if node_type == "class" and isinstance(node, ast.ClassDef) and node.name == name:
                matches.append(node)
            elif node_type == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                matches.append(node)
            elif node_type is None:
                if isinstance(node, ast.ClassDef) and node.name == name:
                    matches.append(node)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    matches.append(node)

        return matches

    def _get_node_end_line(self, node: ast.AST) -> int:
        """
        Get the ending line of a node, with fallback for Python < 3.8.

        Args:
            node: AST node

        Returns:
            Ending line number (1-indexed)
        """
        if hasattr(node, 'end_lineno') and node.end_lineno is not None:
            return node.end_lineno

        # Fallback for older Python versions
        # Walk through all children and find the maximum lineno
        max_line = node.lineno if hasattr(node, 'lineno') else 0

        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                max_line = max(max_line, child.lineno)

        # Try to find the actual end by looking at the source
        if max_line > 0 and max_line <= len(self.lines):
            # Scan forward to find the end of the block
            start_line = node.lineno - 1
            if start_line < len(self.lines):
                base_indent = self._get_indentation_level(
                    self.lines[start_line])

                for i in range(max_line, len(self.lines)):
                    line = self.lines[i]
                    if line.strip():  # Non-empty line
                        line_indent = self._get_indentation_level(line)
                        # If we find a line at same or less indentation, previous line was the end
                        if len(line_indent) <= len(base_indent):
                            return i

                # If we reach here, the block extends to the end
                return len(self.lines)

        return max_line

    def add_code_block(
        self,
        new_code: str,
        location: str = "end_of_file",
        target_name: Optional[str] = None,
        before: bool = False
    ) -> Dict[str, Any]:
        """
        Add a code block (function or class) to the file.

        Args:
            new_code: The code block to add (must be valid Python)
            location: Where to add the code. Options:
                - "end_of_file": At the end of the file
                - "inside_class": Inside a class (requires target_name)
                - "after_function": After a function (requires target_name)
                - "after_class": After a class (requires target_name)
                - "before_function": Before a function (requires target_name)
                - "before_class": Before a class (requires target_name)
                - "beginning_of_file": At the beginning after imports
            target_name: Name of the class/function for relative positioning
            before: If True, add before target instead of after

        Returns:
            Dict with status and modified code
        """
        # Validate the new code
        try:
            new_tree = ast.parse(new_code)
        except SyntaxError as e:
            print("---aborting add_code_block----")  # FIXME delete dbg code
            print(new_code)
            print("Invalid Python syntax in new code: ")
            print("------------\n\n")
            return {
                "success": False,
                "error": f"Invalid Python syntax in new code: {e}",
                "filepath": self.filepath
            }

        # Ensure new code ends with proper newlines
        if not new_code.endswith('\n'):
            new_code += '\n'
        if not new_code.endswith('\n\n'):
            new_code += '\n'

        if location == "end_of_file":
            modified_code = self.source_code.rstrip() + '\n\n\n' + new_code

        elif location == "beginning_of_file":
            # Find the last import
            last_import_line = 0
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if hasattr(node, 'lineno'):
                        last_import_line = max(last_import_line, node.lineno)

            if last_import_line > 0:
                # Add after last import
                lines_before = self.lines[:last_import_line]
                lines_after = self.lines[last_import_line:]
                modified_code = '\n'.join(
                    lines_before) + '\n\n\n' + new_code + '\n'.join(lines_after)
            else:
                # No imports, add at the beginning
                modified_code = new_code + '\n\n' + self.source_code

        elif location == "inside_class":
            if not target_name:
                return {
                    "success": False,
                    "error": "target_name required for 'inside_class' location",
                    "filepath": self.filepath
                }

            # Find the class (now supports nested classes)
            class_nodes = self._find_node_by_name(target_name, "class")

            if not class_nodes:
                return {
                    "success": False,
                    "error": f"Class '{target_name}' not found",
                    "filepath": self.filepath
                }

            if len(class_nodes) > 1:
                return {
                    "success": False,
                    "error": f"Multiple classes named '{target_name}' found. Please be more specific.",
                    "filepath": self.filepath,
                    "warning": "Consider renaming one of the classes for clarity"
                }

            class_node = class_nodes[0]

            # Get the indentation of the class
            class_line = self.lines[class_node.lineno - 1]
            class_indent = self._get_indentation_level(class_line)
            method_indent = class_indent + self.indent_style

            # Properly indent the new code
            indented_code = self._indent_code(new_code, method_indent)

            # Find the best insertion point: after the last method in the class
            last_method_line = class_node.lineno  # Start with class definition line

            for item in class_node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    item_end = self._get_node_end_line(item)
                    last_method_line = max(last_method_line, item_end)

            # Insert after the last method
            insertion_line = last_method_line

            lines_before = self.lines[:insertion_line]
            lines_after = self.lines[insertion_line:]

            # Add proper spacing
            modified_code = '\n'.join(
                lines_before) + '\n\n' + indented_code + '\n' + '\n'.join(lines_after)

        elif location in ["after_function", "after_class", "before_function", "before_class"]:
            if not target_name:
                return {
                    "success": False,
                    "error": f"target_name required for '{location}' location",
                    "filepath": self.filepath
                }

            # Find the target
            is_class = "class" in location
            target_nodes = self._find_node_by_name(
                target_name,
                "class" if is_class else "function"
            )

            if not target_nodes:
                entity_type = "Class" if is_class else "Function"
                return {
                    "success": False,
                    "error": f"{entity_type} '{target_name}' not found",
                    "filepath": self.filepath
                }

            # Filter to only top-level nodes for these operations
            top_level_nodes = [
                node for node in target_nodes if node in self.tree.body]

            if not top_level_nodes:
                entity_type = "Class" if is_class else "Function"
                return {
                    "success": False,
                    "error": f"No top-level {entity_type.lower()} '{target_name}' found (nested definitions not supported for this operation)",
                    "filepath": self.filepath
                }

            if len(top_level_nodes) > 1:
                entity_type = "Class" if is_class else "Function"
                return {
                    "success": False,
                    "error": f"Multiple {entity_type.lower()}s named '{target_name}' found at top level",
                    "filepath": self.filepath,
                    "warning": f"Found {len(top_level_nodes)} definitions. Consider using unique names."
                }

            target_node = top_level_nodes[0]

            if "before" in location or before:
                # Insert before the target
                insertion_line = target_node.lineno - 1

                # Check for decorators
                if hasattr(target_node, 'decorator_list') and target_node.decorator_list:
                    first_decorator = target_node.decorator_list[0]
                    if hasattr(first_decorator, 'lineno'):
                        insertion_line = first_decorator.lineno - 1

                lines_before = self.lines[:insertion_line]
                lines_after = self.lines[insertion_line:]
                modified_code = '\n'.join(
                    lines_before) + '\n\n' + new_code + '\n'.join(lines_after)
            else:
                # Insert after the target
                end_line = self._get_node_end_line(target_node)
                lines_before = self.lines[:end_line]
                lines_after = self.lines[end_line:]
                modified_code = '\n'.join(
                    lines_before) + '\n\n' + new_code + '\n'.join(lines_after)

        else:
            return {
                "success": False,
                "error": f"Unknown location: {location}",
                "filepath": self.filepath
            }

        # Validate the modified code
        try:
            ast.parse(modified_code)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Modified code has syntax error: {e}",
                "filepath": self.filepath
            }

        # Write the modified code
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(modified_code)

        return {
            "success": True,
            "message": f"Code block added at {location}",
            "filepath": self.filepath,
            "location": location,
            "target_name": target_name
        }

    def refactor_rename_symbol(
        self,
        old_name: str,
        new_name: str,
        symbol_type: Optional[Literal["function",
                                      "class", "variable", "all"]] = "all"
    ) -> Dict[str, Any]:
        """
        Safely rename a symbol (function, class, or variable) in the file.

        Args:
            old_name: Current name of the symbol
            new_name: New name for the symbol
            symbol_type: Type of symbol to rename ("function", "class", "variable", "all")

        Returns:
            Dict with status and number of replacements made
        """
        if not new_name.isidentifier():
            return {
                "success": False,
                "error": f"Invalid Python identifier: '{new_name}'",
                "filepath": self.filepath
            }

        # Check for duplicates before renaming
        if symbol_type in ["function", "all"]:
            func_nodes = self._find_node_by_name(old_name, "function")
            if len(func_nodes) > 1:
                return {
                    "success": False,
                    "error": f"Multiple functions named '{old_name}' found ({len(func_nodes)} occurrences)",
                    "filepath": self.filepath,
                    "warning": "Renaming would affect all occurrences. Consider refactoring to use unique names first.",
                    "occurrences": len(func_nodes)
                }

        if symbol_type in ["class", "all"]:
            class_nodes = self._find_node_by_name(old_name, "class")
            if len(class_nodes) > 1:
                return {
                    "success": False,
                    "error": f"Multiple classes named '{old_name}' found ({len(class_nodes)} occurrences)",
                    "filepath": self.filepath,
                    "warning": "Renaming would affect all occurrences. Consider refactoring to use unique names first.",
                    "occurrences": len(class_nodes)
                }

        # Track what we're renaming
        renamed_count = 0
        renamed_locations = []

        # Use ast.NodeTransformer to rename symbols
        class SymbolRenamer(ast.NodeTransformer):
            def __init__(self, old_name, new_name, symbol_type):
                self.old_name = old_name
                self.new_name = new_name
                self.symbol_type = symbol_type
                self.count = 0
                self.locations = []

            def visit_FunctionDef(self, node):
                if (self.symbol_type in ["function", "all"]) and node.name == self.old_name:
                    node.name = self.new_name
                    self.count += 1
                    self.locations.append(f"function at line {node.lineno}")
                self.generic_visit(node)
                return node

            def visit_AsyncFunctionDef(self, node):
                if (self.symbol_type in ["function", "all"]) and node.name == self.old_name:
                    node.name = self.new_name
                    self.count += 1
                    self.locations.append(
                        f"async function at line {node.lineno}")
                self.generic_visit(node)
                return node

            def visit_ClassDef(self, node):
                if (self.symbol_type in ["class", "all"]) and node.name == self.old_name:
                    node.name = self.new_name
                    self.count += 1
                    self.locations.append(f"class at line {node.lineno}")
                self.generic_visit(node)
                return node

            def visit_Name(self, node):
                if (self.symbol_type in ["variable", "all"]) and node.id == self.old_name:
                    node.id = self.new_name
                    self.count += 1
                    if hasattr(node, 'lineno'):
                        self.locations.append(
                            f"variable at line {node.lineno}")
                return node

            def visit_arg(self, node):
                if (self.symbol_type in ["variable", "all"]) and node.arg == self.old_name:
                    node.arg = self.new_name
                    self.count += 1
                    if hasattr(node, 'lineno'):
                        self.locations.append(
                            f"argument at line {node.lineno}")
                return node

        # Apply the renaming
        renamer = SymbolRenamer(old_name, new_name, symbol_type)
        new_tree = renamer.visit(self.tree)

        renamed_count = renamer.count
        renamed_locations = renamer.locations

        if renamed_count == 0:
            return {
                "success": False,
                "error": f"Symbol '{old_name}' not found (type: {symbol_type})",
                "filepath": self.filepath
            }

        # Convert AST back to source code
        try:
            modified_code = ast.unparse(new_tree)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate code from AST: {e}",
                "filepath": self.filepath
            }

        # Validate the modified code
        try:
            ast.parse(modified_code)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Modified code has syntax error: {e}",
                "filepath": self.filepath
            }

        # Write the modified code
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(modified_code)

        return {
            "success": True,
            "message": f"Renamed '{old_name}' to '{new_name}'",
            "filepath": self.filepath,
            "old_name": old_name,
            "new_name": new_name,
            "symbol_type": symbol_type,
            "occurrences": renamed_count,
            "locations": renamed_locations[:10]  # Limit to 10 locations
        }

    def delete_code_block(
        self,
        block_name: str,
        block_type: Optional[Literal["function", "class", "auto"]] = "auto"
    ) -> Dict[str, Any]:
        """
        Delete a function or class by name.

        Args:
            block_name: Name of the function or class to delete
            block_type: Type of block ("function", "class", or "auto" to detect)

        Returns:
            Dict with status and information about deleted block
        """
        # Find the block to delete
        target_nodes = []
        target_type = None

        if block_type in ["function", "auto"]:
            func_nodes = [node for node in self.tree.body
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and node.name == block_name]
            if func_nodes:
                target_nodes.extend(func_nodes)
                target_type = "function"

        if block_type in ["class", "auto"]:
            class_nodes = [node for node in self.tree.body
                           if isinstance(node, ast.ClassDef)
                           and node.name == block_name]
            if class_nodes:
                target_nodes.extend(class_nodes)
                target_type = "class"

        if not target_nodes:
            return {
                "success": False,
                "error": f"Block '{block_name}' not found (type: {block_type})",
                "filepath": self.filepath
            }

        if len(target_nodes) > 1:
            return {
                "success": False,
                "error": f"Multiple blocks named '{block_name}' found ({len(target_nodes)} occurrences)",
                "filepath": self.filepath,
                "warning": "Please specify which one to delete or rename them to be unique"
            }

        target_node = target_nodes[0]

        if isinstance(target_node, ast.AsyncFunctionDef):
            target_type = "async function"

        # Get the line range to delete
        start_line = target_node.lineno - 1  # 0-indexed

        # Check for decorators (they come before the node)
        if hasattr(target_node, 'decorator_list') and target_node.decorator_list:
            first_decorator = target_node.decorator_list[0]
            if hasattr(first_decorator, 'lineno'):
                start_line = first_decorator.lineno - 1

        end_line = self._get_node_end_line(target_node)

        # Delete the lines
        lines_before = self.lines[:start_line]
        lines_after = self.lines[end_line:]

        # Remove extra blank lines
        while lines_before and not lines_before[-1].strip():
            lines_before.pop()
        while lines_after and not lines_after[0].strip():
            lines_after.pop(0)

        modified_code = '\n'.join(lines_before) + \
            '\n\n' + '\n'.join(lines_after)

        # Validate the modified code
        try:
            ast.parse(modified_code)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Modified code has syntax error: {e}",
                "filepath": self.filepath
            }

        # Write the modified code
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(modified_code)

        return {
            "success": True,
            "message": f"Deleted {target_type} '{block_name}'",
            "filepath": self.filepath,
            "block_name": block_name,
            "block_type": target_type,
            "lines_deleted": end_line - start_line
        }


# Utility functions for easy access

def add_code_block(
    filepath: str,
    new_code: str,
    location: str = "end_of_file",
    target_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add a code block to a Python file.

    Args:
        filepath: Path to the Python file
        new_code: Code block to add
        location: Where to add ("end_of_file", "inside_class", "after_function", etc.)
        target_name: Name of target class/function for relative positioning

    Returns:
        Dict with operation result

    Example:
        result = add_code_block(
            "app.py",
            "def delete_user(user_id):\\n    pass",
            location="after_function",
            target_name="create_user"
        )
    """
    refactorer = PythonRefactoring(filepath)
    return refactorer.add_code_block(new_code, location, target_name)


def refactor_rename_symbol(
    filepath: str,
    old_name: str,
    new_name: str,
    symbol_type: str = "all"
) -> Dict[str, Any]:
    """
    Safely rename a symbol in a Python file.

    Args:
        filepath: Path to the Python file
        old_name: Current name
        new_name: New name
        symbol_type: Type of symbol ("function", "class", "variable", "all")

    Returns:
        Dict with operation result

    Example:
        result = refactor_rename_symbol(
            "app.py",
            "calc_ttl",
            "calculate_total",
            symbol_type="function"
        )
    """
    refactorer = PythonRefactoring(filepath)
    return refactorer.refactor_rename_symbol(old_name, new_name, symbol_type)


def delete_code_block(
    filepath: str,
    block_name: str,
    block_type: str = "auto"
) -> Dict[str, Any]:
    """
    Delete a function or class from a Python file.

    Args:
        filepath: Path to the Python file
        block_name: Name of the function/class to delete
        block_type: Type of block ("function", "class", "auto")

    Returns:
        Dict with operation result

    Example:
        result = delete_code_block(
            "app.py",
            "old_api_call",
            block_type="function"
        )
    """
    refactorer = PythonRefactoring(filepath)
    return refactorer.delete_code_block(block_name, block_type)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python python_refactoring.py <test|demo>")
        sys.exit(1)

    if sys.argv[1] == "demo":
        print("Python Refactoring Tools Demo")
        print("=" * 50)
        print("\nAvailable operations:")
        print("1. add_code_block() - Add code at semantic locations")
        print("2. refactor_rename_symbol() - Safe symbol renaming")
        print("3. delete_code_block() - Remove functions/classes")
        print("\nSee tests for usage examples.")
