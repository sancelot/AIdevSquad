"""
Universal refactoring interface - automatically detects language and uses the appropriate tool.
Supports Python, TypeScript, JavaScript, and more.
"""

from .typescript_refactoring import TypeScriptRefactoring
from .python_refactoring import PythonRefactoring
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Import language-specific refactoring tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class UniversalRefactoring:
    """Universal refactoring interface supporting multiple languages."""

    # Language detection based on file extension
    LANGUAGE_MAP = {
        '.py': 'python',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
    }

    def __init__(self, filepath: str):
        """
        Initialize universal refactoring tool.

        Args:
            filepath: Path to the file to refactor
        """
        self.filepath = filepath
        self.extension = Path(filepath).suffix.lower()
        self.language = self.LANGUAGE_MAP.get(self.extension, 'unknown')

        # Initialize the appropriate refactoring tool
        if self.language == 'python':
            self.refactorer = PythonRefactoring(filepath)
        elif self.language in ['typescript', 'javascript']:
            self.refactorer = TypeScriptRefactoring(filepath)
        else:
            self.refactorer = None

    def add_code_block(
        self,
        new_code: str,
        location: str = "end_of_file",
        target_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a code block at a semantic location.

        Args:
            new_code: The code block to add
            location: Semantic location ("end_of_file", "inside_class", "after_function", etc.)
            target_name: Name of target for relative positioning

        Returns:
            Dict with operation result
        """
        if not self.refactorer:
            return {
                "success": False,
                "error": f"Language '{self.language}' not supported yet",
                "filepath": self.filepath,
                "supported_languages": list(set(self.LANGUAGE_MAP.values()))
            }

        result = self.refactorer.add_code_block(
            new_code, location, target_name)
        result['language'] = self.language
        return result

    def refactor_rename_symbol(
        self,
        old_name: str,
        new_name: str,
        symbol_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Safely rename a symbol (function, class, variable).

        Args:
            old_name: Current name
            new_name: New name
            symbol_type: Type of symbol ("function", "class", "variable", "all")

        Returns:
            Dict with operation result
        """
        if not self.refactorer:
            return {
                "success": False,
                "error": f"Language '{self.language}' not supported yet",
                "filepath": self.filepath,
                "supported_languages": list(set(self.LANGUAGE_MAP.values()))
            }

        result = self.refactorer.refactor_rename_symbol(
            old_name, new_name, symbol_type)
        result['language'] = self.language
        return result

    def delete_code_block(
        self,
        block_name: str,
        block_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Delete a code block (function, class, interface).

        Args:
            block_name: Name of the block to delete
            block_type: Type of block ("function", "class", "auto")

        Returns:
            Dict with operation result
        """
        if not self.refactorer:
            return {
                "success": False,
                "error": f"Language '{self.language}' not supported yet",
                "filepath": self.filepath,
                "supported_languages": list(set(self.LANGUAGE_MAP.values()))
            }

        result = self.refactorer.delete_code_block(block_name, block_type)
        result['language'] = self.language
        return result

    @classmethod
    def get_supported_languages(cls) -> list:
        """Get list of supported languages."""
        return sorted(list(set(cls.LANGUAGE_MAP.values())))

    @classmethod
    def get_supported_extensions(cls) -> dict:
        """Get mapping of extensions to languages."""
        return cls.LANGUAGE_MAP.copy()


# Simple API functions

def add_code_block(
    filepath: str,
    new_code: str,
    location: str = "end_of_file",
    target_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add a code block to any supported file.

    Automatically detects the language and uses the appropriate refactoring tool.

    Args:
        filepath: Path to the file
        new_code: Code block to add
        location: Where to add it ("end_of_file", "inside_class", etc.)
        target_name: Target for relative positioning

    Returns:
        Dict with operation result including 'language' field

    Example:
        # Python
        result = add_code_block(
            "app.py",
            "def delete_user(user_id):\\n    pass",
            location="after_function",
            target_name="create_user"
        )

        # TypeScript
        result = add_code_block(
            "api.ts",
            "async function deleteUser(id: string) { }",
            location="end_of_file"
        )
    """
    refactorer = UniversalRefactoring(filepath)
    return refactorer.add_code_block(new_code, location, target_name)


def refactor_rename_symbol(
    filepath: str,
    old_name: str,
    new_name: str,
    symbol_type: str = "all"
) -> Dict[str, Any]:
    """
    Safely rename a symbol in any supported file.

    Args:
        filepath: Path to the file
        old_name: Current name
        new_name: New name
        symbol_type: Type of symbol ("function", "class", "variable", "all")

    Returns:
        Dict with operation result including 'language' field

    Example:
        # Python
        result = refactor_rename_symbol(
            "app.py",
            "calc_ttl",
            "calculate_total"
        )

        # TypeScript
        result = refactor_rename_symbol(
            "api.ts",
            "fetchData",
            "fetchUserData",
            symbol_type="function"
        )
    """
    refactorer = UniversalRefactoring(filepath)
    return refactorer.refactor_rename_symbol(old_name, new_name, symbol_type)


def delete_code_block(
    filepath: str,
    block_name: str,
    block_type: str = "auto"
) -> Dict[str, Any]:
    """
    Delete a code block from any supported file.

    Args:
        filepath: Path to the file
        block_name: Name of the block to delete
        block_type: Type of block ("function", "class", "auto")

    Returns:
        Dict with operation result including 'language' field

    Example:
        # Python
        result = delete_code_block(
            "app.py",
            "old_api_call",
            block_type="function"
        )

        # TypeScript
        result = delete_code_block(
            "api.ts",
            "OldComponent",
            block_type="class"
        )
    """
    refactorer = UniversalRefactoring(filepath)
    return refactorer.delete_code_block(block_name, block_type)


def get_supported_languages() -> list:
    """Get list of all supported programming languages."""
    return UniversalRefactoring.get_supported_languages()


def get_supported_extensions() -> dict:
    """Get mapping of file extensions to languages."""
    return UniversalRefactoring.get_supported_extensions()


if __name__ == "__main__":
    print("🔧 Universal Refactoring Tools")
    print("=" * 60)
    print("\n✨ Semantic code modifications for multiple languages")
    print("\n📋 Supported Languages:")

    for lang in get_supported_languages():
        exts = [ext for ext, l in get_supported_extensions().items()
                if l == lang]
        print(f"  • {lang.capitalize()}: {', '.join(exts)}")

    print("\n🛠️  Available Operations:")
    print("  1. add_code_block() - Add code at semantic locations")
    print("  2. refactor_rename_symbol() - Safe symbol renaming")
    print("  3. delete_code_block() - Remove functions/classes/interfaces")

    print("\n💡 Why These Tools Are Better Than replace_text:")
    print("  ✅ Semantic understanding - uses AST, not text matching")
    print("  ✅ Safe - won't accidentally rename comments or strings")
    print("  ✅ Simple - just give the name, not exact text to match")
    print("  ✅ Multi-language - works across different languages")

    print("\n📖 Example Usage:")
    print("""
    # Add a new function
    result = add_code_block(
        "app.py",
        "def delete_user(user_id):\\n    pass",
        location="after_function",
        target_name="create_user"
    )

    # Rename a function safely
    result = refactor_rename_symbol(
        "app.py",
        "calc_ttl",
        "calculate_total",
        symbol_type="function"
    )

    # Delete an obsolete function
    result = delete_code_block(
        "app.py",
        "old_api_call",
        block_type="function"
    )
    """)

    print("\n📚 See test files for more examples!")
