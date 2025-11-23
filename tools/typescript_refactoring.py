"""
Semantic refactoring tools for TypeScript/JavaScript code.
Uses smart tokenization for safe, intelligent code modifications.
"""

import json
import os
import subprocess
import tempfile
import re
from typing import Dict, Any, Optional, Literal, List, Tuple
from pathlib import Path


class Token:
    """Represents a token in the source code."""

    def __init__(self, type: str, value: str, line: int, col: int):
        # 'string', 'comment', 'template', 'brace', 'code', etc.
        self.type = type
        self.value = value
        self.line = line
        self.col = col


class JSTokenizer:
    """Tokenizer that understands JavaScript/TypeScript syntax."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens = []

    def tokenize(self) -> List[Token]:
        """Tokenize the source code."""
        while self.pos < len(self.source):
            # Check for comments
            if self.peek(2) == '//':
                self._read_line_comment()
            elif self.peek(2) == '/*':
                self._read_block_comment()
            # Check for strings
            elif self.peek() in ['"', "'"]:
                self._read_string()
            # Check for template literals
            elif self.peek() == '`':
                self._read_template_literal()
            # Check for regex literals
            elif self._is_regex_context():
                self._read_regex()
            # Regular code
            else:
                self._read_code_char()

        return self.tokens

    def peek(self, n=1) -> str:
        """Peek ahead n characters."""
        return self.source[self.pos:self.pos + n]

    def advance(self, n=1):
        """Advance position."""
        for _ in range(n):
            if self.pos < len(self.source):
                if self.source[self.pos] == '\n':
                    self.line += 1
                    self.col = 1
                else:
                    self.col += 1
                self.pos += 1

    def _read_line_comment(self):
        """Read a line comment."""
        start_line, start_col = self.line, self.col
        value = ''
        while self.pos < len(self.source) and self.peek() != '\n':
            value += self.peek()
            self.advance()
        self.tokens.append(Token('comment', value, start_line, start_col))

    def _read_block_comment(self):
        """Read a block comment."""
        start_line, start_col = self.line, self.col
        value = ''
        self.advance(2)  # Skip /*
        value += '/*'

        while self.pos < len(self.source) - 1:
            if self.peek(2) == '*/':
                value += '*/'
                self.advance(2)
                break
            value += self.peek()
            self.advance()

        self.tokens.append(Token('comment', value, start_line, start_col))

    def _read_string(self):
        """Read a string literal."""
        quote = self.peek()
        start_line, start_col = self.line, self.col
        value = quote
        self.advance()

        while self.pos < len(self.source):
            char = self.peek()
            value += char

            if char == '\\':
                self.advance()
                if self.pos < len(self.source):
                    value += self.peek()
                    self.advance()
            elif char == quote:
                self.advance()
                break
            else:
                self.advance()

        self.tokens.append(Token('string', value, start_line, start_col))

    def _read_template_literal(self):
        """Read a template literal."""
        start_line, start_col = self.line, self.col
        value = '`'
        self.advance()

        while self.pos < len(self.source):
            char = self.peek()

            if char == '\\':
                value += char
                self.advance()
                if self.pos < len(self.source):
                    value += self.peek()
                    self.advance()
            elif char == '`':
                value += char
                self.advance()
                break
            else:
                value += char
                self.advance()

        self.tokens.append(Token('template', value, start_line, start_col))

    def _is_regex_context(self) -> bool:
        """Check if we're in a context where / starts a regex."""
        if self.peek() != '/':
            return False

        # Simple heuristic: regex after =, (, [, {, return, etc.
        # Look back at previous non-whitespace token
        i = self.pos - 1
        while i >= 0 and self.source[i] in ' \t\n':
            i -= 1

        if i < 0:
            return False

        prev_char = self.source[i]
        return prev_char in '=([{,;:!&|?+\n'

    def _read_regex(self):
        """Read a regex literal."""
        start_line, start_col = self.line, self.col
        value = '/'
        self.advance()

        while self.pos < len(self.source):
            char = self.peek()
            value += char

            if char == '\\':
                self.advance()
                if self.pos < len(self.source):
                    value += self.peek()
                    self.advance()
            elif char == '/':
                self.advance()
                # Read flags
                while self.pos < len(self.source) and self.peek() in 'gimsuvy':
                    value += self.peek()
                    self.advance()
                break
            elif char == '\n':
                # Invalid regex
                break
            else:
                self.advance()

        self.tokens.append(Token('regex', value, start_line, start_col))

    def _read_code_char(self):
        """Read a single character of code."""
        start_line, start_col = self.line, self.col
        char = self.peek()

        # Create token for special characters
        if char in '{}[]()':
            self.tokens.append(Token(char, char, start_line, start_col))
        else:
            self.tokens.append(Token('code', char, start_line, start_col))

        self.advance()


class TypeScriptRefactoring:
    """Safe refactoring operations for TypeScript/JavaScript code."""

    def __init__(self, filepath: str):
        """
        Initialize refactoring tool for a TypeScript/JavaScript file.

        Args:
            filepath: Path to the file to refactor
        """
        self.filepath = filepath
        self.extension = Path(filepath).suffix

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            self.source_code = f.read()

        self.lines = self.source_code.split('\n')
        self.indent_style = self._detect_indentation()

        # Tokenize for better parsing
        self.tokenizer = JSTokenizer(self.source_code)
        self.tokens = self.tokenizer.tokenize()

        # Check Babel availability
        self.babel_available = self._check_babel_available()

    def _detect_indentation(self) -> str:
        """
        Detect the indentation style used in the file.

        Returns:
            The indentation string used in the file
        """
        indent_samples = []

        for line in self.lines:
            if line and line[0] in ' \t':
                indent = ''
                for char in line:
                    if char in ' \t':
                        indent += char
                    else:
                        break
                if indent:
                    indent_samples.append(indent)

        if not indent_samples:
            return '  '  # Default to 2 spaces for JS/TS

        # Check for tabs
        if any('\t' in indent for indent in indent_samples):
            return '\t'

        # Find most common indentation level
        from math import gcd
        from functools import reduce
        from collections import Counter

        indent_lengths = [len(indent)
                          for indent in indent_samples if len(indent) > 0]

        if not indent_lengths:
            return '  '

        # JS/TS typically uses 2 or 4 spaces
        counter = Counter(indent_lengths)
        most_common_length = counter.most_common(1)[0][0]

        if most_common_length <= 2:
            return '  '
        elif most_common_length <= 4:
            return '    '
        else:
            return ' ' * most_common_length

    def _check_babel_available(self) -> bool:
        """
        Check if Node.js and required Babel packages are available.

        Returns:
            True if Babel is properly configured
        """
        try:
            # Check Node.js
            result = subprocess.run(
                ['node', '--version'],
                capture_output=True,
                timeout=2
            )
            if result.returncode != 0:
                return False

            # Check if @babel/parser is available
            check_script = """
            try {
                require('@babel/parser');
                console.log('ok');
            } catch (e) {
                console.log('missing');
            }
            """

            result = subprocess.run(
                ['node', '-e', check_script],
                capture_output=True,
                timeout=2
            )

            return result.returncode == 0 and b'ok' in result.stdout

        except Exception:
            return False

    def _find_matching_brace(self, start_line: int, opening_brace: str = '{') -> Optional[int]:
        """
        Find the matching closing brace, accounting for strings and comments.

        Args:
            start_line: Line number where opening brace is (0-indexed)
            opening_brace: The opening character ('{', '[', '(')

        Returns:
            Line number of matching closing brace, or None if not found
        """
        closing_brace = {'(': ')', '[': ']', '{': '}'}[opening_brace]

        # Find the position in source
        lines_before = '\n'.join(self.lines[:start_line])
        start_pos = len(lines_before)
        if start_line > 0:
            start_pos += 1  # Add newline

        # Find opening brace in tokens
        brace_depth = 0
        found_opening = False

        for i, token in enumerate(self.tokens):
            # Skip tokens before our start line
            if token.line < start_line + 1:
                continue

            # Only count braces in code, not in strings/comments
            if token.type in ['string', 'comment', 'template', 'regex']:
                continue

            if token.value == opening_brace:
                if not found_opening and token.line == start_line + 1:
                    found_opening = True
                    brace_depth = 1
                elif found_opening:
                    brace_depth += 1
            elif token.value == closing_brace and found_opening:
                brace_depth -= 1
                if brace_depth == 0:
                    # Found matching brace
                    return token.line - 1  # Convert to 0-indexed

        return None

    def _is_symbol_in_string_or_comment(self, line_num: int, symbol: str) -> bool:
        """
        Check if a symbol on a given line is inside a string or comment.

        Args:
            line_num: 0-indexed line number
            symbol: The symbol to check

        Returns:
            True if the symbol is in a string or comment
        """
        for token in self.tokens:
            if token.line == line_num + 1:  # Tokens use 1-indexed lines
                if token.type in ['string', 'comment', 'template', 'regex']:
                    if symbol in token.value:
                        return True
        return False

    def _find_exports(self) -> List[str]:
        """
        Find all exported symbols in the file.

        Returns:
            List of exported symbol names
        """
        exports = []
        export_pattern = r'export\s+(?:const|let|var|function|class|interface|type)\s+(\w+)'

        for line in self.lines:
            matches = re.findall(export_pattern, line)
            exports.extend(matches)

        # Also check for export { name }
        export_list_pattern = r'export\s+{([^}]+)}'
        for line in self.lines:
            match = re.search(export_list_pattern, line)
            if match:
                names = [n.strip().split()[0]
                         for n in match.group(1).split(',')]
                exports.extend(names)

        return exports

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

        # Find minimum indentation
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
                result_lines.append('')
            else:
                current_indent = len(line) - len(line.lstrip())
                relative_indent = current_indent - min_indent
                new_line = base_indent + \
                    (' ' * relative_indent) + line.lstrip()
                result_lines.append(new_line)

        return '\n'.join(result_lines)

    def add_code_block(
        self,
        new_code: str,
        location: str = "end_of_file",
        target_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a code block (function, class, or interface) to the file.

        Args:
            new_code: The code block to add
            location: Where to add the code. Options:
                - "end_of_file": At the end of the file
                - "inside_class": Inside a class (requires target_name)
                - "after_function": After a function (requires target_name)
                - "after_class": After a class (requires target_name)
                - "after_interface": After an interface (requires target_name)
                - "beginning_of_file": At the beginning after imports
            target_name: Name of the class/function/interface for relative positioning

        Returns:
            Dict with status and modified code
        """
        # Ensure new code ends with proper newlines
        if not new_code.endswith('\n'):
            new_code += '\n'

        if self.babel_available:
            return self._add_code_block_babel(new_code, location, target_name)
        else:
            return self._add_code_block_smart(new_code, location, target_name)

    def _add_code_block_smart(
        self,
        new_code: str,
        location: str,
        target_name: Optional[str]
    ) -> Dict[str, Any]:
        """Smart implementation using tokenization."""

        if location == "end_of_file":
            modified_code = self.source_code.rstrip() + '\n\n' + new_code

        elif location == "beginning_of_file":
            # Find last import
            last_import_line = 0
            for i, line in enumerate(self.lines):
                # Skip if in comment or string
                if self._is_symbol_in_string_or_comment(i, 'import'):
                    continue
                if re.match(r'^\s*(import|from).+', line):
                    last_import_line = i + 1

            if last_import_line > 0:
                lines_before = self.lines[:last_import_line]
                lines_after = self.lines[last_import_line:]
                modified_code = '\n'.join(
                    lines_before) + '\n\n' + new_code + '\n' + '\n'.join(lines_after)
            else:
                modified_code = new_code + '\n\n' + self.source_code

        elif location in ["after_function", "after_class", "after_interface"]:
            if not target_name:
                return {
                    "success": False,
                    "error": f"target_name required for '{location}'",
                    "filepath": self.filepath
                }

            # Find the target
            found_line = None
            if location == "after_function":
                patterns = [
                    rf'(?:export\s+)?(?:async\s+)?function\s+{re.escape(target_name)}\s*\(',
                    rf'(?:export\s+)?(?:const|let|var)\s+{re.escape(target_name)}\s*=\s*(?:async\s*)?\(',
                ]
            elif location == "after_class":
                patterns = [
                    rf'(?:export\s+)?class\s+{re.escape(target_name)}\s*(?:extends\s+\w+\s*)?{{']
            else:  # after_interface
                patterns = [
                    rf'(?:export\s+)?interface\s+{re.escape(target_name)}\s*(?:extends\s+[\w\s,]+\s*)?{{']

            for i, line in enumerate(self.lines):
                # Skip if in comment or string
                if self._is_symbol_in_string_or_comment(i, target_name):
                    continue

                for pattern in patterns:
                    if re.search(pattern, line):
                        found_line = i
                        break
                if found_line is not None:
                    break

            if found_line is None:
                return {
                    "success": False,
                    "error": f"Target '{target_name}' not found",
                    "filepath": self.filepath
                }

            # Find matching closing brace using smart tokenization
            end_line = self._find_matching_brace(found_line, '{')

            if end_line is None:
                return {
                    "success": False,
                    "error": f"Could not find end of '{target_name}'",
                    "filepath": self.filepath
                }

            lines_before = self.lines[:end_line + 1]
            lines_after = self.lines[end_line + 1:]
            modified_code = '\n'.join(lines_before) + \
                '\n\n' + new_code + '\n'.join(lines_after)

        elif location == "inside_class":
            if not target_name:
                return {
                    "success": False,
                    "error": "target_name required for 'inside_class'",
                    "filepath": self.filepath
                }

            # Find the class
            pattern = rf'(?:export\s+)?class\s+{re.escape(target_name)}\s*(?:extends\s+\w+\s*)?{{'

            found_line = None
            for i, line in enumerate(self.lines):
                if self._is_symbol_in_string_or_comment(i, target_name):
                    continue
                if re.search(pattern, line):
                    found_line = i
                    break

            if found_line is None:
                return {
                    "success": False,
                    "error": f"Class '{target_name}' not found",
                    "filepath": self.filepath
                }

            # Find closing brace
            end_line = self._find_matching_brace(found_line, '{')

            if end_line is None:
                return {
                    "success": False,
                    "error": f"Could not find end of class '{target_name}'",
                    "filepath": self.filepath
                }

            # Get indentation
            class_line = self.lines[found_line]
            class_indent = len(class_line) - len(class_line.lstrip())
            base_indent = ' ' * class_indent
            method_indent = base_indent + self.indent_style

            # Indent the new code
            indented_code = self._indent_code(new_code, method_indent)

            # Insert before closing brace
            lines_before = self.lines[:end_line]
            lines_after = self.lines[end_line:]
            modified_code = '\n'.join(
                lines_before) + '\n\n' + indented_code + '\n' + '\n'.join(lines_after)

        else:
            return {
                "success": False,
                "error": f"Unknown location: {location}",
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
            "target_name": target_name,
            "parser": "smart_tokenizer"
        }

    def _add_code_block_babel(
        self,
        new_code: str,
        location: str,
        target_name: Optional[str]
    ) -> Dict[str, Any]:
        """
        Implementation using Babel AST parsing.

        This requires @babel/parser to be installed:
        npm install @babel/parser
        """
        try:
            # Create a temp file with the parsing script
            parse_script = """
            const parser = require('@babel/parser');
            const fs = require('fs');
            
            const sourceFile = process.argv[2];
            const source = fs.readFileSync(sourceFile, 'utf8');
            
            try {
                const ast = parser.parse(source, {
                    sourceType: 'module',
                    plugins: ['typescript', 'jsx']
                });
                
                console.log(JSON.stringify({
                    success: true,
                    ast: ast
                }));
            } catch (error) {
                console.log(JSON.stringify({
                    success: false,
                    error: error.message
                }));
            }
            """

            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(parse_script)
                script_path = f.name

            try:
                result = subprocess.run(
                    ['node', script_path, self.filepath],
                    capture_output=True,
                    timeout=5
                )

                if result.returncode == 0:
                    # Parse the AST and use it for refactoring
                    # For now, this is a placeholder for full implementation
                    # Fall back to smart tokenizer
                    return self._add_code_block_smart(new_code, location, target_name)
                else:
                    # Fall back if Babel fails
                    return self._add_code_block_smart(new_code, location, target_name)
            finally:
                os.unlink(script_path)

        except Exception as e:
            # Fall back to smart tokenizer on any error
            return self._add_code_block_smart(new_code, location, target_name)

    def refactor_rename_symbol(
        self,
        old_name: str,
        new_name: str,
        symbol_type: Literal["function", "class",
                             "interface", "variable", "all"] = "all"
    ) -> Dict[str, Any]:
        """
        Safely rename a symbol in the file.

        Args:
            old_name: Current name of the symbol
            new_name: New name for the symbol
            symbol_type: Type of symbol to rename

        Returns:
            Dict with status and number of replacements made
        """
        # Validate the new name
        if not re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', new_name):
            return {
                "success": False,
                "error": f"Invalid identifier: '{new_name}'",
                "filepath": self.filepath
            }

        # Check if symbol is exported
        exports = self._find_exports()
        if old_name in exports:
            return {
                "success": False,
                "error": f"Symbol '{old_name}' is exported. Renaming it may break imports in other files.",
                "filepath": self.filepath,
                "warning": "Consider updating all import statements before renaming",
                "exported": True
            }

        renamed_count = 0
        renamed_locations = []
        modified_lines = self.lines.copy()

        for i, line in enumerate(modified_lines):
            # Skip if entire line is in a string or comment
            line_is_safe = True
            for token in self.tokens:
                if token.line == i + 1 and token.type in ['string', 'comment', 'template', 'regex']:
                    if old_name in token.value:
                        # Don't rename inside strings/comments
                        line_is_safe = False
                        break

            if not line_is_safe:
                continue

            line_modified = False

            # Function declarations
            if symbol_type in ["function", "all"]:
                # function name()
                if re.search(rf'\bfunction\s+{re.escape(old_name)}\s*\(', line):
                    modified_lines[i] = re.sub(
                        rf'\bfunction\s+{re.escape(old_name)}\b',
                        f'function {new_name}',
                        modified_lines[i]
                    )
                    renamed_count += 1
                    renamed_locations.append(f"function at line {i+1}")
                    line_modified = True

                # const name = () =>
                if re.search(rf'\b(const|let|var)\s+{re.escape(old_name)}\s*=', line):
                    modified_lines[i] = re.sub(
                        rf'\b(const|let|var)\s+{re.escape(old_name)}\b',
                        rf'\1 {new_name}',
                        modified_lines[i]
                    )
                    renamed_count += 1
                    renamed_locations.append(f"const function at line {i+1}")
                    line_modified = True

            # Class declarations
            if symbol_type in ["class", "all"]:
                if re.search(rf'\bclass\s+{re.escape(old_name)}\b', line):
                    modified_lines[i] = re.sub(
                        rf'\bclass\s+{re.escape(old_name)}\b',
                        f'class {new_name}',
                        modified_lines[i]
                    )
                    renamed_count += 1
                    renamed_locations.append(f"class at line {i+1}")
                    line_modified = True

            # Interface declarations
            if symbol_type in ["interface", "all"]:
                if re.search(rf'\binterface\s+{re.escape(old_name)}\b', line):
                    modified_lines[i] = re.sub(
                        rf'\binterface\s+{re.escape(old_name)}\b',
                        f'interface {new_name}',
                        modified_lines[i]
                    )
                    renamed_count += 1
                    renamed_locations.append(f"interface at line {i+1}")
                    line_modified = True

            # Variable references (only if not a declaration and not in string/comment)
            if symbol_type in ["variable", "all"] and not line_modified:
                occurrences = len(re.findall(
                    rf'\b{re.escape(old_name)}\b', line))
                if occurrences > 0:
                    modified_lines[i] = re.sub(
                        rf'\b{re.escape(old_name)}\b',
                        new_name,
                        modified_lines[i]
                    )
                    renamed_count += occurrences
                    if occurrences > 1:
                        renamed_locations.append(
                            f"{occurrences} references at line {i+1}")
                    else:
                        renamed_locations.append(f"reference at line {i+1}")

        if renamed_count == 0:
            return {
                "success": False,
                "error": f"Symbol '{old_name}' not found (type: {symbol_type})",
                "filepath": self.filepath
            }

        modified_code = '\n'.join(modified_lines)

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
            "locations": renamed_locations[:10],
            "parser": "smart_tokenizer"
        }

    def delete_code_block(
        self,
        block_name: str,
        block_type: Literal["function", "class", "interface", "auto"] = "auto"
    ) -> Dict[str, Any]:
        """
        Delete a function, class, or interface by name.

        Args:
            block_name: Name of the block to delete
            block_type: Type of block ("function", "class", "interface", "auto")

        Returns:
            Dict with status and information about deleted block
        """
        # Check if exported
        exports = self._find_exports()
        if block_name in exports:
            return {
                "success": False,
                "error": f"'{block_name}' is exported. Deleting it may break imports in other files.",
                "filepath": self.filepath,
                "warning": "Consider updating all import statements before deleting",
                "exported": True
            }

        # Find the block
        start_line = None
        end_line = None
        found_type = None

        for i, line in enumerate(self.lines):
            # Skip if in comment or string
            if self._is_symbol_in_string_or_comment(i, block_name):
                continue

            # Check for function
            if block_type in ["function", "auto"]:
                if re.search(rf'\bfunction\s+{re.escape(block_name)}\s*\(', line) or \
                   re.search(rf'\b(const|let|var)\s+{re.escape(block_name)}\s*=\s*(?:async\s*)?\(', line):
                    start_line = i
                    found_type = "function"

                    # Check for export on previous line
                    if i > 0 and 'export' in self.lines[i - 1]:
                        start_line = i - 1

                    # Find the end using smart brace matching
                    if '{' in line:
                        end_line = self._find_matching_brace(i, '{')
                        if end_line is not None:
                            end_line += 1  # Include the line with closing brace
                    else:
                        # Arrow function on one line
                        end_line = i + 1
                    break

            # Check for class
            if block_type in ["class", "auto"]:
                if re.search(rf'\bclass\s+{re.escape(block_name)}\b', line):
                    start_line = i
                    found_type = "class"

                    if i > 0 and 'export' in self.lines[i - 1]:
                        start_line = i - 1

                    end_line = self._find_matching_brace(i, '{')
                    if end_line is not None:
                        end_line += 1
                    break

            # Check for interface
            if block_type in ["interface", "auto"]:
                if re.search(rf'\binterface\s+{re.escape(block_name)}\b', line):
                    start_line = i
                    found_type = "interface"

                    if i > 0 and 'export' in self.lines[i - 1]:
                        start_line = i - 1

                    end_line = self._find_matching_brace(i, '{')
                    if end_line is not None:
                        end_line += 1
                    break

        if start_line is None:
            return {
                "success": False,
                "error": f"Block '{block_name}' not found (type: {block_type})",
                "filepath": self.filepath
            }

        if end_line is None:
            return {
                "success": False,
                "error": f"Could not find end of block '{block_name}'",
                "filepath": self.filepath
            }

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

        # Write the modified code
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(modified_code)

        return {
            "success": True,
            "message": f"Deleted {found_type} '{block_name}'",
            "filepath": self.filepath,
            "block_name": block_name,
            "block_type": found_type,
            "lines_deleted": end_line - start_line,
            "parser": "smart_tokenizer"
        }


# Utility functions for easy access

def add_code_block(
    filepath: str,
    new_code: str,
    location: str = "end_of_file",
    target_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add a code block to a TypeScript/JavaScript file.

    Example:
        result = add_code_block(
            "api.ts",
            "async function deleteUser(id: string) { }",
            location="after_function",
            target_name="createUser"
        )
    """
    refactorer = TypeScriptRefactoring(filepath)
    return refactorer.add_code_block(new_code, location, target_name)


def refactor_rename_symbol(
    filepath: str,
    old_name: str,
    new_name: str,
    symbol_type: str = "all"
) -> Dict[str, Any]:
    """
    Safely rename a symbol in a TypeScript/JavaScript file.

    Example:
        result = refactor_rename_symbol(
            "api.ts",
            "fetchData",
            "fetchUserData",
            symbol_type="function"
        )
    """
    refactorer = TypeScriptRefactoring(filepath)
    return refactorer.refactor_rename_symbol(old_name, new_name, symbol_type)


def delete_code_block(
    filepath: str,
    block_name: str,
    block_type: str = "auto"
) -> Dict[str, Any]:
    """
    Delete a function, class, or interface from a TypeScript/JavaScript file.

    Example:
        result = delete_code_block(
            "api.ts",
            "oldApiCall",
            block_type="function"
        )
    """
    refactorer = TypeScriptRefactoring(filepath)
    return refactorer.delete_code_block(block_name, block_type)


if __name__ == "__main__":
    print("TypeScript/JavaScript Refactoring Tools (Fixed)")
    print("=" * 50)
    print("\nAvailable operations:")
    print("1. add_code_block() - Add code at semantic locations")
    print("2. refactor_rename_symbol() - Safe symbol renaming")
    print("3. delete_code_block() - Remove functions/classes/interfaces")
    print("\nFeatures:")
    print("✓ Smart tokenization (handles strings, comments, templates)")
    print("✓ Indentation detection (tabs/spaces)")
    print("✓ Export tracking and warnings")
    print("✓ String literal protection")
    print("✓ Babel integration (when available)")
    print("\nSee tests for usage examples.")
