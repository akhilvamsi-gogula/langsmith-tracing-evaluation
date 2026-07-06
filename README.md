# Production-Grade LLM Evaluation Suite: LangSmith Implementation

A comprehensive hands-on implementation of **production-grade LLM observability, tracing, and evaluation** using LangSmith, LangChain, and Groq. This project demonstrates real-world patterns for building robust LLM evaluation pipelines.

## 🎯 Project Highlights

This repository showcases practical expertise in:

- **LangSmith Tracing Architecture**: End-to-end telemetry capturing, run introspection, and performance metrics aggregation
- **Custom Evaluator Development**: Designing and implementing multiple evaluation strategies (strict, lenient, multi-metric)
- **Experiment Management**: Automated A/B testing, prompt optimization, and regression analysis
- **Production Quality**: Structured evaluation datasets, versioned experiments, and reproducible benchmarks

## 🏗️ Architecture & Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM Orchestration** | LangChain | Unified interface for composing LLM pipelines with structured inputs/outputs |
| **LLM Provider** | Groq Cloud (Qwen 3.6-27B) | High-throughput open-source model inference |
| **Observability** | LangSmith | Tracing, experiment tracking, evaluator execution, and analytics |
| **Language** | Python 3.x | Core implementation language |

## 📊 Implementation Breakdown

### Phase 1: Observability Foundation
**`task1_tracing.py`** — LangSmith Tracing Setup
- Configure LANGCHAIN_TRACING_V2 and project namespacing
- Instrument LLM calls for automatic trace capture
- **Key learning**: Understand LangSmith's trace payload structure (inputs, outputs, metadata, latency)

### Phase 2: Trace Analytics
**`task2_explore_traces.py`** — Operational Metrics Analysis
- Query LangSmith client API for traces and runs
- Extract and aggregate performance metrics across batches
- Measure latency, token usage, error rates
- **Key learning**: Navigate the LangSmith client API for production monitoring

### Phase 3: Evaluation Framework (Strict Evaluator)
**`task3_exact_match.py`** — Binary Exact-Match Scoring
- Implement evaluator interface: `def evaluator(run, example) -> {"key": str, "score": float}`
- Build case-insensitive, whitespace-normalized comparison logic
- Test against Q&A dataset with mixed correct/incorrect outputs
- **Key learning**: Evaluators are pluggable scoring functions; design for both strictness and edge case handling

### Phase 4: Experiment Orchestration
**`task4_first_experiment.py`** — Dataset Creation & Baseline Runs
- Design test dataset structure (inputs, reference outputs)
- Create LangSmith datasets via client API
- Run evaluation experiments with single evaluator
- **Key learning**: Versioned datasets enable reproducible benchmarking across experiment iterations

### Phase 5: Regression Analysis
**`task5_analyze_results.py`** — A/B Prompt Comparison
- Compare experiment results across different prompts
- Measure impact of prompt engineering (conciseness, accuracy trade-offs)
- Identify statistical significance
- **Key learning**: Data-driven prompt optimization using experiment comparisons

### Phase 6: Evaluation Framework (Lenient Evaluator)
**`task6_contains_evaluator.py`** — Substring-Match Scoring
- Implement partial-match evaluator for natural language robustness
- Compare strict vs. lenient scoring on identical outputs
- Demonstrate evaluator trade-off: precision vs. coverage
- **Key learning**: Real-world LLM evaluation requires multiple metrics to measure different quality dimensions

### Phase 7: Production Evaluation
**`task7_multi_evaluator.py`** — Multi-Metric Experiment Execution
- Run single experiment with multiple evaluators simultaneously
- Generate dual-metric scorecards for comprehensive quality assessment
- Demonstrate end-to-end evaluation pipeline
- **Key learning**: Production systems require multi-metric evaluation to avoid false positives/negatives

## 🔑 Key Concepts Demonstrated

### 1. **LangSmith Tracing**
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "qa-evaltest"

llm = ChatGroq(model="qwen/qwen3.6-27b")
response = llm.invoke([HumanMessage(content="...")])
# Automatically traced to LangSmith
```
- All LLM calls logged with latency, token counts, and structured payloads
- Projects organize traces by logical grouping

### 2. **Evaluator Pattern**
```python
def exact_match_evaluator(run, example):
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    score = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
    return {"key": "exact_match", "score": score}
```
- Evaluators are stateless functions receiving model output + reference
- Scores are floats (0.0-1.0) mapped to human-interpretable metrics
- LangSmith aggregates scores across experiment runs

### 3. **Experiment Management**
```python
results = evaluate(
    qa_pipeline,
    data=dataset_name,
    evaluators=[exact_match_evaluator, contains_evaluator],
    experiment_prefix="qa-evaltest-multi"
)
```
- Experiments execute pipeline on dataset + apply evaluators
- Multi-evaluator experiments capture quality from multiple angles
- LangSmith dashboard provides side-by-side comparison views

## 📈 Quality Metrics Implemented

| Evaluator | Scoring Logic | Use Case |
|-----------|---------------|----------|
| **Exact Match** | `prediction.lower() == reference.lower()` | Factual Q&A where precision is critical |
| **Contains Match** | `reference.lower() in prediction.lower()` | Natural language where verbosity is acceptable |
| **Multi-Metric** | Both scores in single experiment | Comprehensive quality assessment |

**Example**: For question "What is the capital of France?" with reference answer "Paris":
- Output: "Paris" → Exact: ✓ (1.0), Contains: ✓ (1.0)
- Output: "The capital is Paris" → Exact: ✗ (0.0), Contains: ✓ (1.0)
- Output: "London" → Exact: ✗ (0.0), Contains: ✗ (0.0)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- LangSmith API key: https://smith.langchain.com
- Groq API key: https://console.groq.com

### Installation

```bash
git clone https://github.com/akhilvamsi-gogula/langsmith-tracing-evaluation.git
cd langsmith-tracing-evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-groq langsmith python-dotenv
```

### Configuration

```bash
# Create .env file with your API keys
cat > .env << EOF
LANGCHAIN_API_KEY=your_langsmith_api_key
GROQ_API_KEY=your_groq_api_key
EOF
```

### Running Tasks (Sequential)

Each task builds on previous work:

```bash
# 1. Enable tracing
python task1_tracing.py

# 2. Explore traces in LangSmith UI
python task2_explore_traces.py

# 3. Test exact match evaluator
python task3_exact_match.py

# 4. Create dataset and run experiment
python task4_first_experiment.py

# 5. Analyze results across experiments
python task5_analyze_results.py

# 6. Test contains (lenient) evaluator
python task6_contains_evaluator.py

# 7. Run production multi-evaluator suite
python task7_multi_evaluator.py
```

**Monitor progress**: Open https://smith.langchain.com to view traces, datasets, and experiments in real-time.

## 📋 LangSmith Skills Demonstrated

✅ **Tracing & Instrumentation**
- Configure tracing environment variables
- Capture structured run data (inputs, outputs, metadata)
- Query traces via LangSmith client API

✅ **Dataset Management**
- Create versioned datasets with inputs/reference outputs
- Structure data for evaluation experiments
- Track dataset lineage across experiment runs

✅ **Evaluator Development**
- Implement custom scoring functions
- Handle edge cases (whitespace, case sensitivity, partial matches)
- Design multi-metric evaluation strategies

✅ **Experiment Execution**
- Run single and batch evaluations
- Compare experiments for regression analysis
- Generate scorecards and performance reports

✅ **Production Patterns**
- Namespace projects for organizational clarity
- Version experiments for reproducibility
- Implement multi-metric evaluation for comprehensive quality assessment

## 🎓 Learning Outcomes

By reviewing this codebase, QA engineers and AI test engineers will understand:

1. **How to instrument LLM applications** for observability
2. **How to design evaluators** that capture meaningful quality signals
3. **How to structure evaluation experiments** for reproducibility
4. **How to use LangSmith** as the backbone of LLM quality assurance
5. **How to measure prompt effectiveness** through A/B experimentation

## 🔧 Production Considerations

This implementation demonstrates patterns applicable to:
- **Quality gates**: Multi-metric evaluation prevents shipping degraded models
- **Regression detection**: Experiment comparisons catch performance drops
- **Prompt optimization**: Structured benchmarking drives iterative improvement
- **Cost analysis**: Token usage tracking optimizes inference expense
- **SLA monitoring**: Latency metrics inform infrastructure decisions

## 📁 Project Structure

```
langsmith-tracing-evaluation/
├── README.md                          # This file
├── .env.example                       # API key template
├── task1_tracing.py                  # Tracing setup
├── task2_explore_traces.py           # Metrics analysis
├── task3_exact_match.py              # Strict evaluator
├── task4_first_experiment.py         # Dataset & baseline
├── task5_analyze_results.py          # Regression analysis
├── task6_contains_evaluator.py       # Lenient evaluator
├── task7_multi_evaluator.py          # Production pipeline
└── data/                              # Test datasets (generated)
```

## 🔗 Resources

- **LangSmith Docs**: https://docs.smith.langchain.com
- **LangChain Docs**: https://python.langchain.com
- **Groq API**: https://console.groq.com/docs
- **LLM Evaluation Best Practices**: https://smith.langchain.com/docs/evaluation

## 📝 Notes

- Each task is designed to run independently for learning/debugging
- Tasks 1-3 run without external API calls (mock data)
- Tasks 4-7 require LangSmith project setup but include error handling
- All evaluators use case-insensitive matching for robustness

## 💡 Advanced Extensions

Potential enhancements for production deployment:
- Add semantic similarity evaluators (embedding-based matching)
- Implement cost/latency trade-off analysis
- Build automated regression detection with alerts
- Create evaluator composition patterns (AND/OR logic)
- Add human review workflows for edge cases

---

**Author**: [Akhil Vamsi Gogula](https://github.com/akhilvamsi-gogula)  
**Repository**: LangSmith Evaluation & Tracing Implementation  
**Purpose**: Showcase hands-on LLM observability and QA automation expertise
