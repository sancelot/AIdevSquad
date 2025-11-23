"""
Tool to generate a high-level summary of a code file.
Optimized for LLM agents to save tokens.
"""

import ast
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class CodeSummary:
    """Class to generate code summaries."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.extension = Path(filepath).suffix
        
    def get_code_summary(self) -> Dict[str, Any]:
        """
        Main entry point. Analyzes the file and returns a structured summary.
        
        Returns:
            Dict containing the code summary with keys:
            - filepath: file path
            - language: detected language
            - imports: list of imports
            - classes: list of classes with their methods
            - functions: list of functions with their signatures
            - global_variables: list of global variables
            - summary: short text summary
            - token_estimate: estimated number of tokens in the summary
        """
        if not os.path.exists(self.filepath):
            return {
                "error": f"File not found: {self.filepath}",
                "filepath": self.filepath
            }
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return {
                "error": "Unable to read file (encoding issue)",
                "filepath": self.filepath
            }
        except Exception as e:
            return {
                "error": f"Error reading file: {str(e)}",
                "filepath": self.filepath
            }
        
        # Determine language and parse accordingly
        if self.extension == '.py':
            return self._parse_python(content)
        elif self.extension in ['.js', '.jsx', '.ts', '.tsx']:
            return self._parse_javascript_basic(content)
        else:
            return self._parse_generic(content)
    
    def _parse_python(self, content: str) -> Dict[str, Any]:
        """Parse a Python file with AST."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                "error": f"Syntax error in Python file: {str(e)}",
                "filepath": self.filepath,
                "language": "python"
            }
        
        imports = []
        classes = []
        functions = []
        global_variables = []
        
        for node in ast.walk(tree):
            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "name": alias.name,
                        "alias": alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname
                    })
        
        # Analyse de premier niveau uniquement (pas les nested)
        for node in tree.body:
            # Classes
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [self._get_name(base) for base in node.bases],
                    "decorators": [self._get_name(dec) for dec in node.decorator_list],
                    "methods": [],
                    "docstring": ast.get_docstring(node)
                }
                
                # Méthodes de la classe
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                        method_info = self._parse_function(item)
                        class_info["methods"].append(method_info)
                
                classes.append(class_info)
            
            # Fonctions (niveau module)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append(self._parse_function(node))
            
            # Variables globales
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_value = self._get_value_repr(node.value)
                        global_variables.append({
                            "name": target.id,
                            "value_type": type(node.value).__name__,
                            "value_preview": var_value
                        })
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    annotation = self._get_name(node.annotation) if node.annotation else None
                    var_value = self._get_value_repr(node.value) if node.value else None
                    global_variables.append({
                        "name": node.target.id,
                        "type_annotation": annotation,
                        "value_preview": var_value
                    })
        
        # Générer un résumé textuel
        summary_text = self._generate_summary_text(classes, functions, imports, global_variables)
        
        result = {
            "filepath": self.filepath,
            "language": "python",
            "imports": imports,
            "classes": classes,
            "functions": functions,
            "global_variables": global_variables,
            "summary": summary_text,
            "stats": {
                "num_classes": len(classes),
                "num_functions": len(functions),
                "num_imports": len(imports),
                "num_global_vars": len(global_variables)
            }
        }
        
        # Estimation des tokens (approximation: 1 token ≈ 4 caractères)
        result["token_estimate"] = len(json.dumps(result)) // 4
        
        return result
    
    def _parse_function(self, node) -> Dict[str, Any]:
        """Parse an AST function node."""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Arguments
        args = []
        func_args = node.args
        
        # Positional args
        for i, arg in enumerate(func_args.args):
            arg_info = {
                "name": arg.arg,
                "annotation": self._get_name(arg.annotation) if arg.annotation else None,
                "default": None
            }
            # Defaults sont alignés à droite
            defaults_offset = len(func_args.args) - len(func_args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                arg_info["default"] = self._get_value_repr(func_args.defaults[default_idx])
            args.append(arg_info)
        
        # *args
        if func_args.vararg:
            args.append({
                "name": f"*{func_args.vararg.arg}",
                "annotation": self._get_name(func_args.vararg.annotation) if func_args.vararg.annotation else None,
                "default": None
            })
        
        # Keyword-only args
        for i, arg in enumerate(func_args.kwonlyargs):
            arg_info = {
                "name": arg.arg,
                "annotation": self._get_name(arg.annotation) if arg.annotation else None,
                "default": self._get_value_repr(func_args.kw_defaults[i]) if func_args.kw_defaults[i] else None
            }
            args.append(arg_info)
        
        # **kwargs
        if func_args.kwarg:
            args.append({
                "name": f"**{func_args.kwarg.arg}",
                "annotation": self._get_name(func_args.kwarg.annotation) if func_args.kwarg.annotation else None,
                "default": None
            })
        
        return {
            "name": node.name,
            "is_async": is_async,
            "args": args,
            "return_annotation": self._get_name(node.returns) if node.returns else None,
            "decorators": [self._get_name(dec) for dec in node.decorator_list],
            "docstring": ast.get_docstring(node)
        }
    
    def _get_name(self, node) -> str:
        """Extracts the name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._get_name(node.value)
            slice_val = self._get_name(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Index):  # Python < 3.9
            return self._get_name(node.value)
        elif isinstance(node, ast.Tuple):
            elements = [self._get_name(elt) for elt in node.elts]
            return f"({', '.join(elements)})"
        elif isinstance(node, ast.List):
            elements = [self._get_name(elt) for elt in node.elts]
            return f"[{', '.join(elements)}]"
        else:
            return type(node).__name__
    
    def _get_value_repr(self, node) -> Optional[str]:
        """Gets a simplified representation of a value."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value) > 50:
                return f'"{node.value[:50]}..."'
            return repr(node.value)
        elif isinstance(node, ast.List):
            if len(node.elts) == 0:
                return "[]"
            return f"[...{len(node.elts)} items]"
        elif isinstance(node, ast.Dict):
            if len(node.keys) == 0:
                return "{}"
            return f"{{...{len(node.keys)} items}}"
        elif isinstance(node, ast.Call):
            func_name = self._get_name(node.func)
            return f"{func_name}(...)"
        else:
            return f"<{type(node).__name__}>"
    
    def _parse_javascript_basic(self, content: str) -> Dict[str, Any]:
        """Basic parsing for JavaScript/TypeScript (without full AST)."""
        lines = content.split('\n')
        imports = []
        functions = []
        classes = []
        exports = []
        
        for line in lines:
            line = line.strip()
            
            # Imports
            if line.startswith('import '):
                imports.append(line)
            elif line.startswith('const ') or line.startswith('let ') or line.startswith('var '):
                if '=' in line and 'require(' in line:
                    imports.append(line)
            
            # Functions
            if line.startswith('function ') or line.startswith('async function '):
                func_match = line.split('(')[0]
                functions.append(func_match.replace('async', '').replace('function', '').strip())
            
            # Classes
            if line.startswith('class '):
                class_name = line.split('{')[0].replace('class', '').replace('extends', ' extends ').strip()
                classes.append(class_name)
            
            # Exports
            if 'export ' in line:
                exports.append(line)
        
        summary_text = f"JavaScript/TypeScript file with {len(classes)} classes, {len(functions)} functions, {len(imports)} imports"
        
        return {
            "filepath": self.filepath,
            "language": "javascript",
            "imports": imports[:20],  # Limiter pour économiser les tokens
            "classes": classes,
            "functions": functions,
            "exports": exports[:10],
            "summary": summary_text,
            "note": "Basic parsing without full AST - consider using a proper JS parser for detailed analysis",
            "stats": {
                "num_classes": len(classes),
                "num_functions": len(functions),
                "num_imports": len(imports)
            }
        }
    
    def _parse_generic(self, content: str) -> Dict[str, Any]:
        """Generic parsing for other languages."""
        lines = content.split('\n')
        num_lines = len(lines)
        num_non_empty = sum(1 for line in lines if line.strip())
        
        # Détection basique de structures
        keywords = {
            'class': 0,
            'function': 0,
            'def': 0,
            'import': 0,
            'include': 0
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            for keyword in keywords:
                if keyword in line_lower:
                    keywords[keyword] += 1
        
        return {
            "filepath": self.filepath,
            "language": "unknown",
            "extension": self.extension,
            "summary": f"File with {num_lines} lines ({num_non_empty} non-empty)",
            "basic_stats": {
                "total_lines": num_lines,
                "non_empty_lines": num_non_empty,
                "estimated_classes": keywords.get('class', 0),
                "estimated_functions": keywords.get('function', 0) + keywords.get('def', 0),
                "estimated_imports": keywords.get('import', 0) + keywords.get('include', 0)
            },
            "note": "Generic parsing - install language-specific parser for better results"
        }
    
    def _generate_summary_text(self, classes: List, functions: List, 
                               imports: List, global_vars: List) -> str:
        """Generates a concise text summary."""
        parts = []
        
        if classes:
            class_names = [c['name'] for c in classes]
            parts.append(f"{len(classes)} class(es): {', '.join(class_names)}")
        
        if functions:
            func_names = [f['name'] for f in functions]
            if len(func_names) > 5:
                parts.append(f"{len(functions)} function(s) including: {', '.join(func_names[:5])}...")
            else:
                parts.append(f"{len(functions)} function(s): {', '.join(func_names)}")
        
        if imports:
            parts.append(f"{len(imports)} import(s)")
        
        if global_vars:
            parts.append(f"{len(global_vars)} global variable(s)")
        
        if not parts:
            return "Empty file or no structures detected"
        
        return " | ".join(parts)


def get_code_summary(filename: str) -> Dict[str, Any]:
    """
    Main utility function to get a code summary.
    
    Args:
        filename: Chemin vers le fichier à analyser
        
    Returns:
        Dictionnaire contenant le résumé structuré du code
        
    Usage example for an LLM agent:
        >>> summary = get_code_summary("app.py")
        >>> print(summary['summary'])
        >>> for func in summary['functions']:
        >>>     print(f"- {func['name']}({', '.join([a['name'] for a in func['args']])})")
    """
    summarizer = CodeSummary(filename)
    return summarizer.get_code_summary()


def get_code_summary_formatted(filename: str) -> str:
    """
    Returns a formatted summary in readable text for an LLM agent.
    
    Args:
        filename: Chemin vers le fichier à analyser
        
    Returns:
        String formaté avec le résumé du code
    """
    summary = get_code_summary(filename)
    
    if "error" in summary:
        return f"❌ Error: {summary['error']}"
    
    output = []
    output.append(f"📄 File: {summary['filepath']}")
    output.append(f"🔤 Language: {summary['language']}")
    output.append(f"📊 Summary: {summary['summary']}")
    output.append("")
    
    if summary.get('imports'):
        output.append("📦 IMPORTS:")
        for imp in summary['imports'][:10]:  # Limiter à 10 pour économiser les tokens
            if isinstance(imp, dict):
                if imp['type'] == 'import':
                    alias = f" as {imp['alias']}" if imp['alias'] else ""
                    output.append(f"  - import {imp['name']}{alias}")
                else:
                    alias = f" as {imp['alias']}" if imp['alias'] else ""
                    output.append(f"  - from {imp['module']} import {imp['name']}{alias}")
            else:
                output.append(f"  - {imp}")
        if len(summary['imports']) > 10:
            output.append(f"  ... and {len(summary['imports']) - 10} more")
        output.append("")
    
    if summary.get('classes'):
        output.append("🏛️  CLASSES:")
        for cls in summary['classes']:
            bases = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
            output.append(f"  class {cls['name']}{bases}:")
            if cls.get('docstring'):
                output.append(f"    \"\"\"{cls['docstring'][:100]}...\"\"\"")
            for method in cls['methods'][:5]:  # Limiter à 5 méthodes
                args_str = ', '.join([a['name'] for a in method['args']])
                async_prefix = "async " if method.get('is_async') else ""
                output.append(f"    - {async_prefix}def {method['name']}({args_str})")
            if len(cls['methods']) > 5:
                output.append(f"    ... and {len(cls['methods']) - 5} more methods")
        output.append("")
    
    if summary.get('functions'):
        output.append("⚙️  FUNCTIONS:")
        for func in summary['functions']:
            args_str = ', '.join([a['name'] for a in func['args']])
            async_prefix = "async " if func.get('is_async') else ""
            return_annotation = f" -> {func['return_annotation']}" if func.get('return_annotation') else ""
            output.append(f"  - {async_prefix}def {func['name']}({args_str}){return_annotation}")
        output.append("")
    
    if summary.get('global_variables'):
        output.append("🌐 GLOBAL VARIABLES:")
        for var in summary['global_variables'][:5]:
            type_info = var.get('type_annotation', var.get('value_type', ''))
            output.append(f"  - {var['name']}: {type_info}")
        if len(summary['global_variables']) > 5:
            output.append(f"  ... and {len(summary['global_variables']) - 5} more")
        output.append("")
    
    if 'token_estimate' in summary:
        output.append(f"💾 Estimated tokens in summary: ~{summary['token_estimate']}")
    
    return '\n'.join(output)


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python get_code_summary.py <filepath>")
        print("\nExemple:")
        print("  python get_code_summary.py app.py")
        print("  python get_code_summary.py --json app.py")
        sys.exit(1)
    
    show_json = '--json' in sys.argv
    filepath = sys.argv[-1]
    
    if show_json:
        summary = get_code_summary(filepath)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(get_code_summary_formatted(filepath))