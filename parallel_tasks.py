"""
Simple asynchronous parallel execution using LangChain and Ollama models.

This script demonstrates running three tasks (summarize, decide, report) concurrently
using Python's asyncio and LangChain's async capabilities, then generates a final report.
"""

import os
import sys
import asyncio
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents.report_generation_agent import run_report_generation
from agents.doc_ingestion_agent import load_documents, combine_documents

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3.5:latest")
TEMPERATURE = 0.2
ARTIFACTS_DIR = Path("artifacts/")


async def summarize_async(text: str) -> str:
    """
    Async function to summarize text.
    
    Args:
        text: Text to summarize
        
    Returns:
        Summary string
    """
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE
    )
    
    prompt = ChatPromptTemplate.from_template(
        "You are a professional summarizer. Analyze the following text and create a concise summary "
        "with exactly 3 bullet points. Each bullet should capture a key point or insight.\n\n"
        "Text to summarize:\n{text}\n\n"
        "Summary (exactly 3 bullets):"
    )
    
    chain = prompt | llm
    response = await chain.ainvoke({"text": text})
    return response.content.strip()


async def decide_async(text: str) -> str:
    """
    Async function to make a decision based on text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Decision string
    """
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE
    )
    
    prompt = ChatPromptTemplate.from_template(
        "You are a decision-making expert. Analyze the following text and determine:\n"
        "1. Whether any action is required (Yes/No)\n"
        "2. A clear justification for your decision\n"
        "3. Two specific next steps or recommendations\n\n"
        "Text to analyze:\n{text}\n\n"
        "Provide your analysis in the following format:\n"
        "Action Required: [Yes/No]\n"
        "Justification: [Your reasoning]\n"
        "Next Steps:\n"
        "1. [First step]\n"
        "2. [Second step]"
    )
    
    chain = prompt | llm
    response = await chain.ainvoke({"text": text})
    return response.content.strip()


async def report_async(summary: str, decision: str, max_words: int = 500) -> str:
    """
    Async function to generate a report from summary and decision.
    
    Args:
        summary: Summary text
        decision: Decision analysis text
        max_words: Maximum words for the report (default: 500)
        
    Returns:
        Report string
    """
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE
    )
    
    prompt = ChatPromptTemplate.from_template(
        "You are a professional reporter. Convert the following information into "
        "a concise, natural-language report suitable for stakeholders. "
        "The report should be clear, professional, and easy to understand.\n\n"
        "Limit the report to approximately {max_words} words.\n\n"
        "Summary:\n{summary}\n\n"
        "Decision & Analysis:\n{decision}\n\n"
        "Professional Report:"
    )
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "summary": summary,
        "decision": decision,
        "max_words": max_words
    })
    return response.content.strip()


async def main():
    """
    Main async function that runs all three tasks concurrently, then generates a final report.
    """
    # Load input from artifacts directory using doc_ingestion_agent functions
    try:
        print("="*80)
        print("LOADING DOCUMENTS FROM ARTIFACTS")
        print("="*80)
        
        # Load documents using doc_ingestion_agent
        documents = load_documents(ARTIFACTS_DIR)
        
        if not documents:
            print(f"Error: No documents found in {ARTIFACTS_DIR}")
            return
        
        # Combine documents using doc_ingestion_agent
        test_input = combine_documents(documents)
        
        if not test_input:
            print(f"Error: No valid content after combining documents from {ARTIFACTS_DIR}")
            return
        
        # Prepare metadata
        input_metadata = {
            "num_documents": len(documents),
            "total_characters": len(test_input)
        }
        
        print(f"✓ Loaded {input_metadata['num_documents']} document(s)")
        print(f"✓ Total input length: {input_metadata['total_characters']} characters")
        
    except Exception as e:
        print(f"Error loading input from artifacts: {e}")
        return
    
    print("\n" + "="*80)
    print("RUNNING PARALLEL TASKS")
    print("="*80)
    print(f"Input text preview: {test_input[:100]}...")
    print("\nExecuting summarize and decide tasks concurrently...\n")
    
    # Run summarize and decide tasks concurrently (they can run in parallel)
    summary, decision = await asyncio.gather(
        summarize_async(test_input),
        decide_async(test_input)
    )
    
    print("\nGenerating report from summary and decision...\n")
    
    # Generate report using the results from summarize and decide
    report = await report_async(summary, decision, max_words=500)
    
    # Print results
    print("="*80)
    print("RESULTS FROM PARALLEL TASKS")
    print("="*80)
    
    print("\n[SUMMARY]")
    print("-" * 80)
    print(summary)
    
    print("\n[DECISION]")
    print("-" * 80)
    print(decision)
    
    print("\n[REPORT]")
    print("-" * 80)
    print(report)
    
    # Generate final structured report using report generation agent
    print("\n" + "="*80)
    print("GENERATING FINAL STRUCTURED REPORT")
    print("="*80)
    
    # Prepare ingestion metadata from loaded files
    ingestion_metadata = {
        "num_documents": input_metadata["num_documents"],
        "total_characters": input_metadata["total_characters"],
        "num_chunks": 1
    }
    
    # Run report generation (synchronous function, but called after async tasks)
    report_result = run_report_generation(
        summary=summary,
        decision=decision,
        reporter_output=report,
        ingestion_metadata=ingestion_metadata,
        output_dir=Path("."),
        output_filename="parallel_tasks_report.txt",
        json_filename="parallel_tasks_report.json"
    )
    
    if report_result.success:
        print("\n✓ Final report generated successfully!")
        print(f"  Text report: {report_result.report_path}")
        print(f"  JSON report: {report_result.json_path}")
    else:
        print(f"\n✗ Report generation failed: {report_result.error_message}")
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

