#!/usr/bin/env python3
"""
Task 5: Analyze and Compare Experiment Results
Run a second experiment with an improved prompt and compare in LangSmith.

Experiments let you quantify the impact of prompt changes - no more guessing.
LangSmith automatically tracks latency, token usage, cost, and evaluator scores.
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
    """Exact match evaluator (from Task 3)."""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    score = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
    return {"key": "exact_match", "score": score}


def main():
    print("="*60)
    print("Task 5: Analyze and Compare Experiment Results")
    print("="*60)

    client = Client()
    llm = ChatGroq(model="qwen/qwen3.6-27b")

    dataset_name = "qa-evaltest-dataset"

    # Verify V1 experiment exists
    print("\nChecking V1 experiment from Task 4...")
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"  [OK] Dataset found: {dataset_name}")
    except Exception:
        print(f"  [ERROR] Dataset '{dataset_name}' not found. Run Task 4 first.")
        return

    print("\n  Your V1 experiment results are in the LangSmith dashboard:")
    print("  https://smith.langchain.com → Datasets → qa-evaltest-dataset → Experiments")

    # ----------------------------------------------------------
    # 1: Define an improved Q&A pipeline
    #
    # Now create a better system prompt that encourages the LLM to give concise answers.
    #
    # Think about what made V1 fail:
    #   - The LLM added extra words ("The answer is Paris" instead of "Paris")
    #   - The LLM gave full sentences instead of short answers
    # ----------------------------------------------------------
    def improved_pipeline(inputs: dict) -> dict:
        question = inputs["question"]
        response = llm.invoke([
            SystemMessage(content="You are a factual Q&A assistant. Answer with ONLY the exact answer. No sentences, no explanations. Examples: 'Paris', '42', 'H2O'. Be as concise as possible."),
            HumanMessage(content=question)
        ])
        return {"output": response.content}

    # ----------------------------------------------------------
    # 2: Run the V2 experiment with evaluate()
    #
    # Use evaluate() with the improved pipeline and the same
    # dataset and evaluator from Task 4.
    # ----------------------------------------------------------
    print("\nRunning improved experiment (V2)...")

    results_v2 = evaluate(
        improved_pipeline,
        data=dataset_name,
        evaluators=[exact_match_evaluator],
        experiment_prefix="qa-evaltest-v2"
    )

    # Display comparison guidance
    print("\n" + "="*60)
    print("V2 Experiment Complete!")
    print("="*60)
    print("\nNow compare V1 and V2 in the LangSmith dashboard:")
    print("   https://smith.langchain.com")
    print("\n  1. Click 'Datasets' in the left sidebar")
    print("  2. Click on 'qa-evaltest-dataset'")
    print("  3. Go to the 'Experiments' tab")
    print("  4. Select both V1 and V2 experiments using checkboxes")
    print("  5. See exact match scores side-by-side")
    print("\nDid the improved prompt score higher on exact match?")

    print("\n" + "="*60)
    print("[OK] Task 5 Complete: Experiment comparison done!")
    print("="*60)


if __name__ == "__main__":
    main()
