**Autonomous Data Analysis Agent**



This personal project makes an agent that autonomously explores MLB bat tracking data (2024 to 2025 Statcast metrics) to identify standout hitters, unusual swing profiles, and relationships between bat speed and performance outcomes, using Claude's tool use and agentic workflow capabilities.



An AI agent autonomously investigates a dataset the way a human analyst would, by forming hypotheses, deciding what to check next based on what it finds, and producing a written summary of its findings, without a fixed, hardcoded analysis script.





**Tools I Used**

* LLM tool use / function calling: defining a set of callable Python functions as tools the model can invoke, with structured JSON schemas describing when and how to use each one
* Agentic loop design: an iterative loop where the model chooses which tool to call, receives the result, and decides its next action based on that result, continuing autonomously across multiple turns rather than answering a single fixed prompt
* API integration: working directly with the Anthropic API (multi-turn conversation state, message roles, tool result handling) rather than a no-code wrapper
* Data analysis with Pandas: descriptive statistics, correlation analysis, and ranked leaderboards computed on demand as the agent requests them
* Defensive software design: input validation and graceful error handling in every tool function, so a bad or ambiguous input (an invalid column name, an ambiguous name match) returns a clear error the agent can recover from, rather than crashing the program
* Reproducible environment setup: virtual environment, .env-based secret management, and a pinned requirements.txt





**How It Works**

* tools.py defines a small set of data analysis functions (lookup, comparison, ranking, correlation, summary statistics) that operate on the dataset
* agent.py defines JSON tool schemas describing those functions to Claude, then runs a loop:
* Send Claude the task and available tools
* Claude decides whether to call a tool
* If it does, the tool runs locally and the result is sent back to Claude
* Claude uses that result to decide its next step
* This repeats until Claude concludes it has enough information, or a turn limit is reached as a safety stop
* The full reasoning trace (every tool call and result) and the agent's final written summary are saved to output\_report.md





**Tech Stack**

Python, Anthropic Claude API, Pandas, python-dotenv





**Example Output**

Given an open-ended prompt to investigate the dataset, the agent independently computed summary statistics and correlations across several metrics, identified which relationships were strongest, cross-referenced individual data points to sanity-check its own findings, and produced a written summary highlighting the most notable patterns and the specific data points that supported them, without being told which comparisons to run in advance.



See output\_report.md for a full example run, including the complete tool-call trace.





**Possible Extensions**

A get\_correlation\_matrix tool to reduce repeated single-pair correlation calls

A lightweight sklearn model to predict an outcome variable from other metrics, letting the agent report on model performance as part of its investigation

Swapping in a different dataset entirely to confirm the agent and tool structure generalize beyond this specific use case

