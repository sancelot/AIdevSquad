
from .base_agent import BaseAgent

class UnitTestAgent(BaseAgent):
    def __init__(self, llm_class, llm_args):
        super().__init__(llm_class, llm_args, temperature=0.1, agent_name="UnitTestAgent")
    def write_tests(self, filename, file_content,logger):
        """
        Generates unit tests for a given file using the pytest framework.
        """
        system_message = f"""You are an expert Software Development Engineer in Test (SDET). Your specialty is writing clear, effective, and comprehensive unit tests using the `pytest` framework for Python.

You will be given the content of a Python file (`{filename}`). Your task is to write a corresponding test file named `test_{filename}`.

**Your testing philosophy:**
-   Focus on testing the public-facing functions/classes.
-   Test for expected success cases.
-   Test for edge cases (e.g., empty inputs, invalid data).
-   Test for expected failure cases (e.g., functions that should raise an error).
-   Use `pytest.fixture` for any necessary setup (like creating a test client for a Flask app).
-   If the code interacts with a database, you must mock the database session or use an in-memory database to ensure tests are isolated and fast. For a Flask-SQLAlchemy app, this often involves setting up a test client and an application context.

Generate only the Python code for the test file. Do not include any other text, explanations, or Markdown formatting.
"""

        prompt = f"""
Here is the content of the file `{filename}` to be tested:
```python
{file_content}

Now, generate the complete code for test_{filename}.
"""
        test_code_content = self._call_llm(system_message, prompt,logger)
        return test_code_content