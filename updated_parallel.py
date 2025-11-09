"""
Optimized parallel execution using LangChain RunnableParallel pattern.

Uses RunnableParallel to run multiple LLM chains concurrently in a clean, declarative way.
Supports different models for different tasks (summarizer, decision, reporter).
"""

import os
import sys
import time
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from agents.report_generation_agent import run_report_generation
from agents.doc_ingestion_agent import load_documents, combine_documents

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
ARTIFACTS_DIR = Path("artifacts/")

# Model configuration - can use different models for different tasks
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "phi3.5:latest")
DECISION_MODEL = os.getenv("DECISION_MODEL", "phi3.5:latest")
REPORTER_MODEL = os.getenv("REPORTER_MODEL", "phi3.5:latest")

# ====================================================================================
# 1️⃣ Define prompts and chains
# ====================================================================================

# Create prompt templates
SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    "You are a professional summarizer. Analyze the following text and create a concise summary "
    "with exactly 3 bullet points. Each bullet should capture a key point or insight.\n\n"
    "Text to summarize:\n{text}\n\n"
    "Summary (exactly 3 bullets):"
)

DECIDE_PROMPT = ChatPromptTemplate.from_template(
    "You are a decision-making expert. Analyze the following text and determine:\n"
    "1. Whether any action is required (Yes/No)\n"
    "2. A clear justification for your decision\n"
    "3. Two specific next steps or recommendations\n\n"
    "Text to analyze:\n{text}\n\n"
    "Provide your analysis in this format:\n"
    "Action Required: [Yes/No]\n"
    "Justification: [Your reasoning]\n"
    "Next Steps:\n1. [First step]\n2. [Second step]"
)

REPORT_PROMPT = ChatPromptTemplate.from_template(
    "You are a professional reporter. Convert the following information into "
    "a concise, natural-language report suitable for stakeholders. "
    "The report should be clear, professional, and easy to understand.\n\n"
    "Limit the report to approximately {max_words} words.\n\n"
    "Summary:\n{summary}\n\n"
    "Decision & Analysis:\n{decision}\n\n"
    "Professional Report:"
)

# Create LLM instances for each task (can use different models)
def create_summarizer_llm() -> ChatOllama:
    return ChatOllama(
        model=SUMMARIZER_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )

def create_decision_llm() -> ChatOllama:
    return ChatOllama(
        model=DECISION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )

def create_reporter_llm() -> ChatOllama:
    return ChatOllama(
        model=REPORTER_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )

# Create chains: prompt | llm
def create_summarize_chain():
    return SUMMARIZE_PROMPT | create_summarizer_llm()

def create_decide_chain():
    return DECIDE_PROMPT | create_decision_llm()

def create_report_chain():
    return REPORT_PROMPT | create_reporter_llm()

# ====================================================================================
# 2️⃣ Create RunnableParallel workflow
# ====================================================================================

def create_parallel_workflow():
    """
    Create a RunnableParallel workflow that runs summarize and decide in parallel.
    
    Returns:
        RunnableParallel that takes {"text": str} and returns {"summary": str, "decision": str}
    """
    return RunnableParallel({
        "summary": create_summarize_chain(),
        "decision": create_decide_chain()
        # "report": create_report_chain(),
    })

# ====================================================================================
# 3️⃣ Main orchestration
# ====================================================================================

def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def main():
    # Start overall timing
    start_time = time.time()
    
    # Step 1: Load and combine documents
    print("="*80)
    print("LOADING DOCUMENTS FROM ARTIFACTS")
    print("="*80)
    step_start = time.time()

    try:
        documents = load_documents(ARTIFACTS_DIR)
        if not documents:
            print(f"Error: No documents found in {ARTIFACTS_DIR}")
            return

        test_input = combine_documents(documents)
        if not test_input:
            print("Error: Combined content is empty.")
            return

        step_duration = time.time() - step_start
        print(f"✓ Loaded {len(documents)} documents.")
        print(f"✓ Total characters: {len(test_input)}")
        print(f"⏱️  Document loading took: {format_duration(step_duration)}\n")

    except Exception as e:
        print(f"Error during document loading: {e}")
        return

    # Step 2: Create parallel workflow
    print("="*80)
    print("INITIALIZING PARALLEL WORKFLOW")
    print("="*80)
    step_start = time.time()
    
    print(f"Summarizer model: {SUMMARIZER_MODEL}")
    print(f"Decision model: {DECISION_MODEL}")
    print(f"Reporter model: {REPORTER_MODEL}")
    
    parallel_workflow = create_parallel_workflow()
    
    step_duration = time.time() - step_start
    print(f"⏱️  Workflow initialization took: {format_duration(step_duration)}\n")

    # Step 3: Run summarize + decide in parallel using RunnableParallel
    print("="*80)
    print("RUNNING PARALLEL TASKS (Summarize + Decide)")
    print("="*80)
    print(f"Input text preview: {test_input[:100]}...\n")
    step_start = time.time()
    
    try:
        # Invoke the parallel workflow - runs both chains concurrently
        results = parallel_workflow.invoke({"text": test_input})
        
        # Extract results
        summary = results["summary"].content.strip() if hasattr(results["summary"], "content") else str(results["summary"]).strip()
        decision = results["decision"].content.strip() if hasattr(results["decision"], "content") else str(results["decision"]).strip()
        
        step_duration = time.time() - step_start
        print("✓ Summary and Decision tasks completed!")
        print(f"⏱️  Parallel execution took: {format_duration(step_duration)}\n")
        
    except Exception as e:
        print(f"Error during parallel execution: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Generate report using the results
    print("="*80)
    print("GENERATING REPORT")
    print("="*80)
    step_start = time.time()
    
    try:
        report_chain = create_report_chain()
        report_result = report_chain.invoke({
            "summary": summary,
            "decision": decision,
            "max_words": 500,
        })
        report = report_result.content.strip() if hasattr(report_result, "content") else str(report_result).strip()
        
        step_duration = time.time() - step_start
        print("✓ Report generated!")
        print(f"⏱️  Report generation took: {format_duration(step_duration)}\n")
        
    except Exception as e:
        print(f"Error during report generation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Structured final report generation
    print("="*80)
    print("GENERATING FINAL STRUCTURED REPORT")
    print("="*80)
    step_start = time.time()
    
    ingestion_metadata = {
        "num_documents": len(documents),
        "total_characters": len(test_input),
        "num_chunks": 1,
    }

    result = run_report_generation(
        summary=summary,
        decision=decision,
        reporter_output=report,
        ingestion_metadata=ingestion_metadata,
        output_dir=Path("."),
        output_filename="parallel_tasks_report.txt",
        json_filename="parallel_tasks_report.json"
    )

    step_duration = time.time() - step_start
    if result.success:
        print("✓ Final report generated successfully!")
        print(f"  Text report: {result.report_path}")
        print(f"  JSON report: {result.json_path}")
        print(f"⏱️  Final report generation took: {format_duration(step_duration)}")
    else:
        print(f"✗ Report generation failed: {result.error_message}")

    # Calculate and display total execution time
    total_duration = time.time() - start_time
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print(f"⏱️  Total execution time: {format_duration(total_duration)}")
    print(f"⏱️  Total execution time: {total_duration:.2f} seconds")
    print("="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("="*80)

# ====================================================================================
# 4️⃣ Entry point
# ====================================================================================

if __name__ == "__main__":
    main()
