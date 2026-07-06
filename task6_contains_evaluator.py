#!/usr/bin/env python3
"""
Task 6: Build a Contains Evaluator
Build a less strict evaluator that checks if the reference answer appears in the output.

In Task 3, you saw exact match fail when the LLM said "The capital of France is Paris"
but the reference was just "Paris". A contains evaluator handles this by checking if
the reference appears anywhere in the output.
"""

import os
import json


def exact_match_evaluator(run, example):
    """Exact match evaluator (from Task 3)."""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    score = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
    return {"key": "exact_match", "score": score}


def contains_evaluator(run, example):
    """
    Evaluate whether the reference answer appears anywhere in the model's output.

    This is more lenient than exact match. If the LLM says
    "The answer is Paris" and the reference is "Paris", this scores 1.0.
    """
    # ----------------------------------------------------------
    # 1: Extract prediction and reference
    #
    # Get the model's output and the reference answer.
    # ----------------------------------------------------------
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")

    # ----------------------------------------------------------
    # 2: Check if reference is contained in the prediction
    #
    # Use case-insensitive comparison. Return 1.0 if the reference
    # appears anywhere in the prediction, 0.0 otherwise.
    # ----------------------------------------------------------
    score = 1.0 if reference.strip().lower() in prediction.strip().lower() else 0.0

    return {"key": "contains_match", "score": score}


def main():
    print("="*60)
    print("Task 6: Build a Contains Evaluator")
    print("="*60)

    # Mock objects for testing
    class MockRun:
        def __init__(self, outputs):
            self.outputs = outputs

    class MockExample:
        def __init__(self, outputs):
            self.outputs = outputs

    # Test cases showing the difference between exact and contains
    test_cases = [
        {
            "output": "Paris",
            "answer": "Paris",
            "description": "Direct answer"
        },
        {
            "output": "The capital of France is Paris",
            "answer": "Paris",
            "description": "Answer in a sentence"
        },
        {
            "output": "Shakespeare wrote Romeo and Juliet",
            "answer": "William Shakespeare",
            "description": "Partial name match"
        },
        {
            "output": "100 degrees Celsius",
            "answer": "100",
            "description": "Number with extra text"
        },
        {
            "output": "London",
            "answer": "Paris",
            "description": "Completely wrong"
        }
    ]

    print("\nComparing Exact Match vs Contains:\n")
    print(f"  {'Test':<30} {'Exact':<8} {'Contains':<10}")
    print("  " + "-"*48)

    exact_total = 0
    contains_total = 0

    for test in test_cases:
        run = MockRun({"output": test["output"]})
        ex = MockExample({"answer": test["answer"]})

        exact = exact_match_evaluator(run, ex)
        contains = contains_evaluator(run, ex)

        exact_icon = "PASS" if exact["score"] == 1.0 else "FAIL"
        contains_icon = "PASS" if contains["score"] == 1.0 else "FAIL"

        print(f"  {test['description']:<30} {exact_icon:<8} {contains_icon:<10}")
        print(f"    Output: \"{test['output']}\"")
        print(f"    Reference: \"{test['answer']}\"")
        print()

        exact_total += exact["score"]
        contains_total += contains["score"]

    print(f"  Exact Match Score:    {int(exact_total)}/{len(test_cases)}")
    print(f"  Contains Match Score: {int(contains_total)}/{len(test_cases)}")
    print(f"\n  Contains is more lenient -- it catches correct answers")
    print(f"  even when the LLM adds extra words around them.")

    print("\n" + "="*60)
    print("[OK] Task 6 Complete: Contains evaluator built!")
    print("="*60)


if __name__ == "__main__":
    main()
