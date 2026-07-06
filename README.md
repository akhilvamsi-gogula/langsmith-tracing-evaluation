# LLM Observability & Evaluation Pipeline with LangSmith

A production-ready pipeline demonstrating LLM observability, automated tracing, and systematic prompt evaluation. This project implements custom strict and lenient evaluators to test LLM accuracy across prompt iterations using open-source models.

## 🛠️ Tech Stack & Architecture
* **Orchestration:** LangChain
* **LLM Provider:** Groq Cloud (Qwen 3.6) for optimized open-source inference
* **Observability & Evaluation:** LangSmith

## 📋 Project Milestones

| Phase | Core Component | Key Engineering Focus |
| :--- | :--- | :--- |
| **Task 1** | Automated Tracing | Setup telemetry to capture latency, token usage, and LLM payloads. |
| **Task 2** | Trace Exploration | Analyze operational metrics across concurrent model invocations. |
| **Task 3** | Exact Match Evaluator | Build strict, case-insensitive evaluation scoring filters. |
| **Task 4** | Experimentation | Create evaluation datasets and run automated test suites. |
| **Task 5** | Regression Analysis | Compare A/B prompt performance to measure conciseness. |
| **Task 6** | Partial Match Evaluator | Design a lenient contains-match evaluator for natural language. |
| **Task 7** | Multi-Metric Eval | Execute multi-evaluator scoring runs for production benchmarking. |

## 🚀 Local Installation & Setup

1. **Clone and Navigate:**
   ```bash
   git clone [https://github.com/akhilvamsi-gogula/langsmith-tracing-evaluation.git](https://github.com/akhilvamsi-gogula/langsmith-tracing-evaluation.git)
   cd langsmith-tracing-evaluation