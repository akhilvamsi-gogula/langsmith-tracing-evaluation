#!/usr/bin/env python3
"""
Task 3: Build an Exact Match Evaluator
Learn how evaluators work in LangSmith by building a simple exact match scorer.

Evaluators are functions that take a model's output and a reference answer,
then return a score. Exact match is the simplest: either the output matches
the reference (score = 1.0) or it does not (score = 0.0).
"""

import os
import json


def exact_match_evaluator(run, example):
    """
    Evaluate whether the model's output exactly matches the reference answer.

    This is a LangSmith-compatible evaluator. It receives:
    - run: Contains the model's outputs (run.outputs)
    - example: Contains the reference outputs (example.outputs)

    Returns a dict with "key" (metric name) and "score" (0.0 or 1.0).
    """
    # ----------------------------------------------------------
    # 1: Extract the model's answer from the run outputs
    #
    # The model's response is stored in run.outputs with the key "output".
    # ----------------------------------------------------------
    prediction = run.outputs.get("output", "")

    # ----------------------------------------------------------
    # 2: Extract the reference answer from the example
    #
    # The expected answer is stored in example.outputs with the key "answer".
    # ----------------------------------------------------------
    reference = example.outputs.get("answer", "")

    # ----------------------------------------------------------
    # 3: Compare and return a score
    #
    # Compare the prediction and reference using case-insensitive matching.
    # Return 1.0 if they match, 0.0 if they don't.
    # ----------------------------------------------------------
    score = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

    return {"key": "exact_match", "score": score}


def main():
    print("="*60)
    print("Task 3: Build an Exact Match Evaluator")
    print("="*60)

    # Test with simple mock objects before using with LangSmith
    class MockRun:
        def __init__(self, outputs):
            self.outputs = outputs

    class MockExample:
        def __init__(self, outputs):
            self.outputs = outputs

    test_cases = [
        {
            "run": MockRun({"output": "Paris"}),
            "example": MockExample({"answer": "Paris"}),
            "expected_score": 1.0,
            "description": "Exact match"
        },
        {
            "run": MockRun({"output": "paris"}),
            "example": MockExample({"answer": "Paris"}),
            "expected_score": 1.0,
            "description": "Case-insensitive match"
        },
        {
            "run": MockRun({"output": "  Paris  "}),
            "example": MockExample({"answer": "Paris"}),
            "expected_score": 1.0,
            "description": "Whitespace-trimmed match"
        },
        {
            "run": MockRun({"output": "The capital of France is Paris"}),
            "example": MockExample({"answer": "Paris"}),
            "expected_score": 0.0,
            "description": "Partial match (should fail exact match)"
        },
        {
            "run": MockRun({"output": "London"}),
            "example": MockExample({"answer": "Paris"}),
            "expected_score": 0.0,
            "description": "Wrong answer"
        }
    ]

    print("\nRunning evaluator tests...\n")

    all_passed = True
    for i, test in enumerate(test_cases, 1):
        result = exact_match_evaluator(test["run"], test["example"])
        passed = result["score"] == test["expected_score"]
        status = "[OK]" if passed else "[FAIL]"

        print(f"  Test {i}: {test['description']}")
        print(f"    Prediction: \"{test['run'].outputs['output']}\"")
        print(f"    Reference:  \"{test['example'].outputs['answer']}\"")
        print(f"    Score: {result['score']} (expected: {test['expected_score']}) {status}")
        print()

        if not passed:
            all_passed = False

    # Run evaluator on the actual dataset
    print("="*60)
    print("Testing evaluator on Q&A dataset:")
    print("="*60)

    data_path = os.path.join("data", "qa_dataset.json")
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            dataset = json.load(f)

        # Simulated model outputs (mix of correct and incorrect)
        simulated_outputs = [
            "Paris",                    # Correct
            "60",                       # Correct
            "Mercury",                  # Correct
            "H2O",                      # Correct
            "Shakespeare",              # Wrong (missing "William")
            "100 degrees Celsius",      # Wrong (extra text)
            "7",                        # Correct
            "CO2",                      # Wrong
            "Pacific Ocean",            # Correct
            "1945"                      # Correct
        ]

        correct = 0
        total = len(dataset)

        for i, (example, output) in enumerate(zip(dataset, simulated_outputs)):
            result = exact_match_evaluator(
                MockRun({"output": output}),
                MockExample(example["outputs"])
            )
            status = "MATCH" if result["score"] == 1.0 else "MISS"
            print(f"  Q: {example['inputs']['question']}")
            print(f"    Expected: \"{example['outputs']['answer']}\" | Got: \"{output}\" | {status}")
            if result["score"] == 1.0:
                correct += 1

        accuracy = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.0f}%)")
        print(f"\nNotice: Exact match is strict -- even partial matches score 0.")
        print(f"In later labs, you will build more flexible evaluators.")

    if all_passed:

        print("\n" + "="*60)
        print("[OK] Task 3 Complete: Exact match evaluator is working!")
        print("="*60)
    else:
        print("\n[ERROR] Some tests failed. Fix the TODOs and try again.")


if __name__ == "__main__":
    main()
