# Autonomous AI Software Development Team (AiDevSquad)

This project implements a multi-agent AI system designed to autonomously develop, review, test, and document software projects from a high-level user prompt. It simulates a team of AI agents, each with a specialized role, working together to deliver a complete, version-controlled software application.

## 🚀 Features

- **Multi-Agent Architecture:** Simulates a real development team with specialized agents:

  - `SADTSARTPlannerAgent`: The primary planner that uses SADT/SART methodology to generate hierarchical workplans and atomic actions.
  - `ProductOwnerAgent`: Acts as the vision holder and requirements consultant, answering agent questions to clarify ambiguity.
  - `DeveloperAgent`: Executes tasks using a ReAct (Reason+Act) model and a suite of tools.
  - `CodeReviewerAgent`: Analyzes the entire codebase for integration issues, bugs, and inconsistencies.
  - `UnitTestAgent`: Generates `pytest` unit tests for the generated code.
  - `DocumentationAgent`: Writes the final project `README.md` file.

- **Flexible Agent Collaboration:** Agents can now communicate dynamically using the `call_agent` tool. For example, a `DeveloperAgent` can ask the `CodeReviewerAgent` for immediate feedback on a specific file, breaking the rigid sequential handoff structure.
- **Tool-Based Actions (ReAct):** The `DeveloperAgent` uses a set of tools (`list_files`, `read_file`, `write_file`, `call_agent`, etc.) to interact with the codebase and other agents.
- **Automated Version Control:** Every completed task is automatically committed to a Git repository, providing a complete, auditable history of the AI's work.
- **RAG-Powered Context:** Uses Retrieval-Augmented Generation (RAG) via LlamaIndex to provide agents with relevant code context, overcoming LLM context window limitations.
- **Pluggable LLM Providers:** Easily switch between local models (via Ollama) and cloud-based APIs (like Google Gemini).
- **Cost & Usage Tracking:** Monitors token usage and estimates costs for each LLM call.
- **Structured Trace Logging:** Generates detailed Markdown and JSON execution traces for easy debugging and monitoring.

## 🏛️ System Architecture

The system is orchestrated by a central `Orchestrator` class that manages the project lifecycle through several distinct phases:

1.  **Planning:** The `SADTSARTPlannerAgent` analyzes the requirement and generates a detailed hierarchical plan (SADT/SART). The `ProductOwnerAgent` is available for consultation throughout the project.
2.  **Development & Collaboration:** The `Orchestrator` iterates through each task in the plan. The `DeveloperAgent` executes the task using its tools. Crucially, the **DeveloperAgent is responsible for quality assurance**. It can use the `call_agent` tool to request ad-hoc reviews or advice from the `CodeReviewerAgent` or `ProductOwnerAgent` at any time during the task.
3.  **Unit Testing (Optional):** If enabled, the `UnitTestAgent` generates `pytest` files for the Python code.
4.  **Final Validation:** The `TesterAgent` runs the test suite (if any) and attempts to start the main application to ensure it's executable.
5.  **Documentation:** If all previous phases succeed, the `DocumentationAgent` generates a `README.md` file for the newly created project.



## 🛠️ Setup & Installation

Follow these steps to set up and run the project.

### Prerequisites

- Python 3.11+
- Git
- Docker (for the `TesterAgent`)
- (Optional) [Ollama](https://ollama.com/) for running local models.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sancelot/AIdevSquad
    cd AIdevSquad
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Set up LLM providers:**
    - Create a `.env` file in the root directory by copying the example:
      ```bash
      cp .env.example .env
      ```
    - **For Google Gemini:**
      - Go to [Google AI Studio](https://aistudio.google.com/) to get an API key.
      - Add it to your `.env` file: `GOOGLE_API_KEY="your-api-key"`
    - **For Ollama (Local Models):**
      - Install [Ollama](https://ollama.com/).
      - Pull the models you want to use. We recommend:
        ```bash
        ollama pull llama3:8b
        ollama pull bge-base-en-v1.5  # For embeddings
        ```

## ▶️ How to Run

The main script is `orchestrator.py`. You can control its behavior using command-line flags.

**Basic Run (using the default provider, e.g., Ollama):**

```bash
python orchestrator.py myproject --prompt myrequirements.txt
```

**Run with Ollama:**

```bash
python orchestrator.py myproject --ollama --prompt myrequirements.txt
```

**Start a new project from scratch (deletes the existing workspace):**

```bash
python orchestrator.py myproject --new --prompt myrequirements.txt

**Generate unit tests as part of the process:**

```bash
python orchestrator.py myproject--with-tests --prompt myrequirements.txt
```

_Flags can be combined:_

```bash
python orchestrator.py myproject --ollama --new --with-tests --prompt myrequirements.txt
```

After each run, a detailed trace log is saved in the `logs/` directory, and the generated project (with its own Git repository) is available in the `..._workspace/` directory.

## Monitor the Results with the UI

A simple Flask-based UI is available to visualize the execution traces.

1.  **Navigate to the UI directory:**

    ```bash
    cd monitoring_ui
    ```

2.  **Install its dependencies (if any):**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the UI application:**

    ```bash
    flask run --port=5001
    ```

4.  Open your browser to **`http://127.0.0.1:5001`** to see the dashboard.

**New Feature: Nested Collaboration View**
The trace view now supports visualizing nested agent calls. When a `DeveloperAgent` consults a `CodeReviewerAgent` or `ProductOwnerAgent` via chat, the sub-agent's thoughts and tool usage are displayed with indentation and distinct styling, providing a clear "call tree" of the collaboration.

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request. Areas for improvement include:

- Providing consistence documentation between agents in order to avoid producing a project that finally builds but is not usable.
- Adding more tools for the `DeveloperAgent` (e.g., AST-based refactoring for specific languages).
- Improving the intelligence of the `CodeReviewerAgent`.
- Expanding the `TesterAgent` to run more complex integration tests.
- Enhancing the monitoring UI.
-

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
