#!/usr/bin/env python3
"""
Task 4: Run Your First Evaluation Experiment
Create a LangSmith dataset, define a target function, and run evaluate().

An experiment = dataset + target function + evaluators.
LangSmith runs your function on every example, applies evaluators, and collects scores.
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
    print("Task 4: Run Your First Evaluation Experiment")
    print("="*60)

    # Initialize client
    client = Client()
    llm = ChatGroq(model="qwen/qwen3.6-27b")

    # Load the dataset
    data_path = os.path.join("data", "qa_dataset.json")
    with open(data_path, "r") as f:
        qa_data = json.load(f)

    print(f"\nLoaded {len(qa_data)} examples from dataset")

    # ----------------------------------------------------------
    # 1: Create a LangSmith dataset
    #
    # Use the client to create a dataset and add examples to it.
    # Each example has inputs (question) and outputs (answer).
    #
    # Steps:
    #   1. Create dataset with client.create_dataset()
    #   2. Loop through qa_data and add each example
    # ----------------------------------------------------------
    dataset_name = "qa-evaltest-dataset"

    # Delete existing dataset if present (for re-runs)
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        client.delete_dataset(dataset_id=existing.id)
        print(f"  Removed existing dataset: {dataset_name}")
    except Exception:
        pass

    # 1: Create the dataset and add examples
    dataset = client.create_dataset(dataset_name=dataset_name, description="Q&A evaluation dataset with 10 factual questions")

    for example in qa_data:
        client.create_example(
            inputs=example["inputs"],
            outputs=example["outputs"],
            dataset_id=dataset.id
        )

    print(f"  Created dataset: {dataset_name} with {len(qa_data)} examples")
    print(f"  View it at: https://smith.langchain.com")

    # ----------------------------------------------------------
    # 2: Define the target function (Q&A pipeline)
    #
    # This is the function that will be evaluated. It takes inputs
    # (a dict with "question") and returns outputs (a dict with "output").
    #
    # The function should:
    #   1. Get the question from inputs
    #   2. Send it to the LLM with a system prompt
    #   3. Return the response
    # ----------------------------------------------------------
    def qa_pipeline(inputs: dict) -> dict:
        question = inputs["question"]
        response = llm.invoke([
            SystemMessage(content="Answer the question in as few words as possible. Give only the answer, no explanation."), 
            HumanMessage(content=question)
        ])
        return {"output": response.content}

    # ----------------------------------------------------------
    # 3: Run the evaluation experiment
    #
    # Use langsmith evaluate() to run the pipeline on every example
    # in the dataset and score with the exact match evaluator.
    # ----------------------------------------------------------
    print("\nRunning evaluation experiment...")

    results = evaluate(
        qa_pipeline, 
        data=dataset_name,
        evaluators=[exact_match_evaluator],
        experiment_prefix="qa-evaltest-v1"
    )

    # Display results
    print("\n" + "="*60)
    print("Experiment Results:")
    print("="*60)
    print(f"  Experiment: qa-evaltest-v1")
    print(f"  Dataset: {dataset_name}")
    print(f"  Evaluator: exact_match")
    print(f"\nView detailed results in your LangSmith dashboard:")
    print(f"   https://smith.langchain.com")
    print("  - See individual scores per example")
    print("  - View input/output pairs")
    print("  - Explore latency and token usage")

    print("\n" + "="*60)
    print("[OK] Task 4 Complete: First experiment is done!")
    print("="*60)


if __name__ == "__main__":
    main()
