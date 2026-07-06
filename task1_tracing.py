#!/usr/bin/env python3
"""
Task 1: Enable LangSmith Tracing
Configure LangSmith tracing to observe  LLM calls.

Tracing is the foundation of evaluation - you cannot evaluate what you cannot see.
LangSmith tracing captures every LLM call: inputs, outputs, latency, and token usage.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from your .env file

# ============================================================
# CONFIGURATION (pre-configured - do not modify)
# ============================================================
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


def main():
    print("="*60)
    print("Task 1: Enable LangSmith Tracing")
    print("="*60)

    # ----------------------------------------------------------
    # 1: Set the LangSmith project name

    # Set the LANGCHAIN_PROJECT environment variable to: "qa-evaltest"
    # ----------------------------------------------------------
    os.environ["LANGCHAIN_PROJECT"] = "qa-evaltest"  

    print(f"\nProject name set to: {os.environ.get('LANGCHAIN_PROJECT')}")
    print(f"Tracing enabled: {os.environ.get('LANGCHAIN_TRACING_V2')}")

    # ----------------------------------------------------------
    # 2: Create a traced LLM call
    #
    # Use LangChain's ChatGroq to make an LLM call.
    # Because tracing is enabled, this call will automatically
    # be logged to your LangSmith dashboard.
    #
    # Create a ChatGroq instance and invoke it with a question.
    # ----------------------------------------------------------
    llm = ChatGroq(model="qwen/qwen3.6-27b")
    response = llm.invoke([HumanMessage(content="What is LangSmith used for?")])

    print(f"\nQuestion: What is LangSmith used for?")
    print(f"Response: {response.content[:200]}...")
    print(f"\nTrace logged to LangSmith project: \"{os.environ.get('LANGCHAIN_PROJECT')}\"")
    print("Check your LangSmith dashboard to see the full trace!")
    print("   https://smith.langchain.com")

    print("\n" + "="*60)
    print("[OK] Task 1 Complete: LangSmith tracing is configured!")
    print("="*60)


if __name__ == "__main__":
    main()
