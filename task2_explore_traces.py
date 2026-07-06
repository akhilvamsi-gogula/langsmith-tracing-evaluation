#!/usr/bin/env python3
"""
Task 2: Explore Traces in LangSmith
Generate multiple Q&A interactions and explore what LangSmith captures for each call.

Each trace is a record of exactly what your LLM did - the raw material for evaluation.
LangSmith captures: input, output, latency, token count, and model used.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# Ensure tracing is enabled
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "qa-evaltest")


def main():
    print("="*60)
    print("Task 2: Explore Traces in LangSmith")
    print("="*60)

    # Initialize the LLM
    llm = ChatGroq(model="qwen/qwen3.6-27b")

    # Sample questions to generate traces
    questions = [
        "What is the capital of France?",
        "What is 15 multiplied by 4?",
        "Who wrote Romeo and Juliet?",
        "What is the boiling point of water in Celsius?",
        "What is the largest ocean on Earth?"
    ]

    # ----------------------------------------------------------
    # 1: Loop through questions and send each to the LLM
    #
    # For each question in the list above:
    #   1. Send it to the LLM using llm.invoke()
    #   2. Print the question and response
    #
    # Each call will automatically create a trace in LangSmith
    # because tracing is enabled.
    # ----------------------------------------------------------
    print(f"\nGenerating traces for {len(questions)} questions...\n")

    for i, question in enumerate(questions, 1):
        print(f"--- Question {i} of {len(questions)} ---")
        # 1: Send question to LLM and print the response
        response = llm.invoke([HumanMessage(content=question)])
        print(f"Q: {question}")
        print(f"A: {response.content[:150]}")
        print()

    print("="*60)
    print("What LangSmith Captured For Each Trace:")
    print("="*60)
    print("  - Input: The question you sent")
    print("  - Output: The LLM's response")
    print("  - Latency: How long the call took")
    print("  - Token Count: Prompt + completion tokens used")
    print("  - Model: Which model processed the request")
    print("\nOpen your LangSmith dashboard to explore these traces!")
    print("   https://smith.langchain.com")
    print(f"\nLook under project: \"{os.environ.get('LANGCHAIN_PROJECT')}\"")

    print("\n" + "="*60)
    print(f"[OK] Task 2 Complete: {len(questions)} traces generated!")
    print("="*60)


if __name__ == "__main__":
    main()
