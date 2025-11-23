import docker
import os
import json

from .base_agent import BaseAgent

from .utils import parse_json_from_response

class TesterAgent(BaseAgent):
    def __init__(self,llm_class, llm_args,work_dir):
        # The tester needs to be precise and follow instructions, so a low temperature is good.
        super().__init__(llm_class, llm_args, temperature=0.1, agent_name="TesterAgent")
        self.work_dir = os.path.abspath(work_dir)
        self.docker_client = docker.from_env()

    def generate_test_plan(self, technical_architecture, logger):
        """
        This is the new "thinking" step. The agent analyzes the architecture
        and decides which commands to run for linting and testing.
        """
        system_message = """You are an expert DevOps and Quality Assurance Engineer. Your job is to create a testing and validation plan for a software project based on its technical architecture.

You will be given the project's architecture, including the technology stack, dependencies, and file structure.

**Your Task:**
Based on the architecture, define the necessary commands for linting and running tests in a sandboxed Docker environment.

**OUTPUT FORMAT:**
Your response MUST be a single, valid JSON object with the following keys:
-   `"docker_base_image"`: The most appropriate base Docker image (e.g., "python:3.9-slim", "node:18-alpine").
-   `"install_command"`: The shell command to install dependencies inside the container (e.g., "pip install -r requirements.txt", "npm install").
-   `"lint_command"`: The shell command to run the linter. If no linter is obvious for the stack, provide an empty string. (e.g., "flake8 .", "npx eslint .").
-   `"test_command"`: The shell command to run the unit tests. If no test framework is obvious, provide an empty string. (e.g., "pytest", "npm test").

**Example for Python/Flask:**
{
  "docker_base_image": "python:3.9-slim",
  "install_command": "pip install -r requirements.txt flake8 pytest",
  "lint_command": "flake8 --ignore=E501,W503 --max-line-length=88 .",
  "test_command": "pytest"
}

**Example for TypeScript/Express:**
{
  "docker_base_image": "node:18-alpine",
  "install_command": "npm install",
  "lint_command": "npx eslint src/**/*.ts",
  "test_command": "npm test"
}
"""
        prompt = f"""
Here is the technical architecture of the project to be tested:
```json
{json.dumps(technical_architecture, indent=2)}
Please generate the test and validation plan in the specified JSON format.
"""
        response_str = self._call_llm(system_message, prompt, logger)
        data = parse_json_from_response(response_str)
        if data and all(k in data for k in ["docker_base_image", "install_command", "lint_command", "test_command"]):
            return data
        else:
            print("❌ Error: TesterAgent failed to generate a valid test plan.")
            return None
    
    def _run_docker_command(self, base_image, install_cmd, run_cmd):
        """A generic helper to build and run a command in a Docker container."""
        if not run_cmd:
            return True, "Command is empty, skipping." # Success if there's nothing to run

        dockerfile_content = f"""
FROM {base_image}
WORKDIR /app
COPY . .
RUN {install_cmd if install_cmd else "echo 'No install command'"}
CMD {json.dumps(run_cmd.split())}
"""
        dockerfile_path = os.path.join(self.work_dir, f"Dockerfile.test-runner")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)

        try:
            image, _ = self.docker_client.images.build(path=self.work_dir, dockerfile=dockerfile_path, tag="test-runner", rm=True)
            # Run the container and wait for it to finish
            container = self.docker_client.containers.run("test-runner", detach=False, remove=True)
            output = container.decode('utf-8')
            print(f"✅ Docker command '{run_cmd}' completed successfully.")
            print(f"--- Output ---\n{output}\n------------")
            return True, output
        except docker.errors.ContainerError as e:
            # The command failed (e.g., linter found issues, tests failed)
            error_output = e.stderr.decode('utf-8')
            print(f"❌ Docker command '{run_cmd}' failed with exit code {e.exit_status}.")
            return False, error_output
        except Exception as e:
            return False, f"An unexpected error occurred during Docker execution: {e}"


    def run_quality_gate(self, technical_architecture, logger):
        """The main entry point for the tester. It thinks, then acts."""
        self.logger = logger
        self.logger.log_phase("Quality Assurance Gate")

        # 1. THINK: Generate the test plan
        self.logger.log_task(1, 1, "Generating QA Plan")
        self.logger.start_step(1, 1)
        test_plan = self.generate_test_plan(technical_architecture, logger)
        if not test_plan:
            return False, "Could not generate a test plan."

        # 2. ACT: Execute the plan
        base_image = test_plan["docker_base_image"]
        install_cmd = test_plan["install_command"]

        # Run Linter
        self.logger.log_task(1, 2, "Executing Linter")
        self.logger.start_step(1, 1)
        lint_success, lint_output = self._run_docker_command(base_image, install_cmd, test_plan["lint_command"])
        if not lint_success:
            return False, f"Linting failed:\n{lint_output}"
        
        # Run Unit Tests
        self.logger.log_task(2, 2, "Executing Unit Tests")
        self.logger.start_step(1, 1)
        test_success, test_output = self._run_docker_command(base_image, install_cmd, test_plan["test_command"])
        if not test_success:
            return False, f"Unit tests failed:\n{test_output}"

        # If we get here, all automated checks passed.
        return True, "All automated quality checks (Linting & Unit Tests) passed."
    
  
        
    