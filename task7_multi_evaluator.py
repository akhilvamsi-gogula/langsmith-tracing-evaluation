#!/usr/bin/env python3
"""
Task 7: Run a Multi-Evaluator Experiment
Run a single experiment with BOTH evaluators to see how the same outputs
score differently under exact match vs contains match.

This is how production evaluation works - you track multiple metrics
simultaneously to get a complete picture of quality.
"""

import os
import json
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

# Ensure tracing is enabled
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "qa-evaltest")


def exact_match_evaluator(run, example):
    """Exact match: 1.0 only if output matches reference exactly."""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    score = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
    return {"key": "exact_match", "score": score}


def contains_evaluator(run, example):
    """Contains match: 1.0 if reference appears anywhere in the output."""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    score = 1.0 if reference.strip().lower() in prediction.strip().lower() else 0.0
    return {"key": "contains_match", "score": score}


def main():
    print("="*60)
    print("Task 7: Run a Multi-Evaluator Experiment")
    print("="*60)

    client = Client()
    llm = ChatGroq(model="qwen/qwen3.6-27b")

    dataset_name = "qa-evaltest-dataset"

    # Verify dataset exists
    try:
        client.read_dataset(dataset_name=dataset_name)
        print(f"\n  Using existing dataset: {dataset_name}")
    except Exception:
        print(f"\n  [ERROR] Dataset '{dataset_name}' not found!")
        print(f"  Run Task 4 first to create it.")
        return

    # ----------------------------------------------------------
    # 1: Define a Q&A pipeline with a clear, concise prompt
    #
    # Create a pipeline that produces concise factual answers.
    # ----------------------------------------------------------
    def qa_pipeline(inputs: dict) -> dict:
        question = inputs["question"]
        response = llm.invoke([
            SystemMessage(content="You are a factual Q&A assistant. Answer with ONLY the exact answer. No sentences, no explanations. Examples: 'Paris', '42', 'H2O'."), 
            HumanMessage(content=question)
        ])
        return {"output": response.content}

    # ----------------------------------------------------------
    # 2: Run evaluate() with BOTH evaluators
    #
    # Pass both exact_match_evaluator and contains_evaluator
    # in the evaluators list.
    # ----------------------------------------------------------
    print("\nRunning multi-evaluator experiment...")

    results = evaluate(
        qa_pipeline,
        data=dataset_name,
        evaluators= [exact_match_evaluator, contains_evaluator],
        experiment_prefix="qa-evaltest-multi"
    )

    print("\n" + "="*60)
    print("Multi-Evaluator Experiment Complete!")
    print("="*60)
    print("\nOpen your LangSmith dashboard to see both metrics:")
    print("   https://smith.langchain.com")
    print("\nGo to Datasets → qa-evaltest-dataset → experiment 'qa-evaltest-multi'")
    print("You will see TWO score columns:")
    print("  - exact_match: Strict - output must match reference exactly")
    print("  - contains_match: Lenient - reference just needs to appear in output")
    print("\nThe gap between these scores shows how often your LLM gives")
    print("correct answers with extra words around them.")

    print("\n" + "="*60)
    print("[OK] Task 7 Complete: Multi-evaluator experiment done!")
    print("="*60)
    print("\nCongratulations! You have completed the LangSmith Evaluation Lab.")
    print("You now know how to:")
    print("  - Configure LangSmith tracing")
    print("  - Build custom evaluators (exact match and contains)")
    print("  - Run evaluation experiments with multiple metrics")
    print("  - Compare experiments to measure prompt impact")


if __name__ == "__main__":
    main()
