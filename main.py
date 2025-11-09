"""
Agentic Workflow Controller
Orchestrates the execution of all agents in the workflow pipeline.

Workflow:
1. Document Ingestion Agent → loads, cleans, and chunks documents
2. Summarizer Agent → generates concise summaries
3. Decision Agent → performs logical analysis and derives insights
4. Reporter Agent → converts decisions into natural explanations
5. Report Generation Agent → compiles final structured report

Requires: Ollama (daemon running + model pulled), LangChain, langchain-ollama
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple

from logging_config import get_logger

# Import all agents
from agents import (
    run_doc_ingestion,
    run_summarizer,
    run_decision,
    run_reporter,
    run_report_generation,
    DocIngestionResult,
    SummarizerResult,
    DecisionResult,
    ReporterResult,
    ReportGenerationResult
)

logger = get_logger("main")

# -------- CONFIG --------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DATA_DIR = Path(os.getenv("DATA_DIR", "artifacts/"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "."))
REPORT_FILENAME = os.getenv("REPORT_FILENAME", "report.txt")
JSON_REPORT_FILENAME = os.getenv("JSON_REPORT_FILENAME", "report.json")

# Agent-specific model configuration (can use different models for different tasks)
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", OLLAMA_MODEL)
DECISION_MODEL = os.getenv("DECISION_MODEL", OLLAMA_MODEL)
REPORTER_MODEL = os.getenv("REPORTER_MODEL", OLLAMA_MODEL)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2.5s", "1m 30s", "2h 15m")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def run_workflow_with_timing(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    summarizer_model: Optional[str] = None,
    decision_model: Optional[str] = None,
    reporter_model: Optional[str] = None,
    temperature: float = 0.2,
    report_filename: Optional[str] = None,
    json_filename: Optional[str] = None
) -> Tuple[bool, float]:
    """
    Run the complete agentic workflow pipeline with timing.
    
    Args:
        data_dir: Directory containing input text files (default: DATA_DIR)
        output_dir: Directory for output reports (default: OUTPUT_DIR)
        summarizer_model: Model name for summarizer agent (default: SUMMARIZER_MODEL)
        decision_model: Model name for decision agent (default: DECISION_MODEL)
        reporter_model: Model name for reporter agent (default: REPORTER_MODEL)
        temperature: Temperature setting for all agents (default: 0.2)
        report_filename: Name of the text report file (default: REPORT_FILENAME)
        json_filename: Name of the JSON report file (default: JSON_REPORT_FILENAME)
        
    Returns:
        Tuple of (success: bool, duration: float) where duration is in seconds
    """
    start_time = time.time()
    
    logger.info("Starting workflow timing...")
    print("\n" + "="*80)
    print("WORKFLOW TIMING STARTED")
    print("="*80)
    
    success = run_workflow(
        data_dir=data_dir,
        output_dir=output_dir,
        summarizer_model=summarizer_model,
        decision_model=decision_model,
        reporter_model=reporter_model,
        temperature=temperature,
        report_filename=report_filename,
        json_filename=json_filename
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Log and print timing information
    logger.info("="*80)
    logger.info(f"WORKFLOW TIMING: Total duration = {format_duration(duration)} ({duration:.2f} seconds)")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("WORKFLOW TIMING SUMMARY")
    print("="*80)
    print(f"Total Processing Time: {format_duration(duration)}")
    print(f"Duration (seconds): {duration:.2f}s")
    print("="*80)
    
    return success, duration


def run_workflow(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    summarizer_model: Optional[str] = None,
    decision_model: Optional[str] = None,
    reporter_model: Optional[str] = None,
    temperature: float = 0.2,
    report_filename: Optional[str] = None,
    json_filename: Optional[str] = None
) -> bool:
    """
    Run the complete agentic workflow pipeline.
    
    Args:
        data_dir: Directory containing input text files (default: DATA_DIR)
        output_dir: Directory for output reports (default: OUTPUT_DIR)
        summarizer_model: Model name for summarizer agent (default: SUMMARIZER_MODEL)
        decision_model: Model name for decision agent (default: DECISION_MODEL)
        reporter_model: Model name for reporter agent (default: REPORTER_MODEL)
        temperature: Temperature setting for all agents (default: 0.2)
        report_filename: Name of the text report file (default: REPORT_FILENAME)
        json_filename: Name of the JSON report file (default: JSON_REPORT_FILENAME)
        
    Returns:
        True if workflow completed successfully, False otherwise
    """
    logger.info("="*80)
    logger.info("AGENTIC WORKFLOW PIPELINE")
    logger.info("="*80)
    logger.info(f"Data directory: {data_dir or DATA_DIR}")
    logger.info(f"Output directory: {output_dir or OUTPUT_DIR}")
    logger.info(f"Summarizer model: {summarizer_model or SUMMARIZER_MODEL}")
    logger.info(f"Decision model: {decision_model or DECISION_MODEL}")
    logger.info(f"Reporter model: {reporter_model or REPORTER_MODEL}")
    logger.info(f"Temperature: {temperature}")
    
    # Use defaults if not provided
    data_dir = data_dir or DATA_DIR
    output_dir = output_dir or OUTPUT_DIR
    summarizer_model = summarizer_model or SUMMARIZER_MODEL
    decision_model = decision_model or DECISION_MODEL
    reporter_model = reporter_model or REPORTER_MODEL
    
    try:
        # Step 1: Document Ingestion Agent
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DOCUMENT INGESTION AGENT")
        logger.info("="*80)
        ingestion_result: DocIngestionResult = run_doc_ingestion(
            data_dir=data_dir,
            chunk_size=1000,
            chunk_overlap=200,
            enable_cleaning=True
        )
        
        if not ingestion_result.success:
            logger.error(f"Document ingestion failed: {ingestion_result.error_message}")
            return False
        
        if not ingestion_result.combined_text:
            logger.error("No text content available after ingestion")
            return False
        
        logger.info(f"✓ Document ingestion completed: {ingestion_result.metadata['num_documents']} documents processed")
        
        # Step 2: Summarizer Agent
        logger.info("\n" + "="*80)
        logger.info("STEP 2: SUMMARIZER AGENT")
        logger.info("="*80)
        summarizer_result: SummarizerResult = run_summarizer(
            text=ingestion_result.combined_text,
            model_name=summarizer_model,
            temperature=temperature,
            num_bullets=3
        )
        
        if not summarizer_result.success:
            logger.error(f"Summarizer agent failed: {summarizer_result.error_message}")
            return False
        
        logger.info(f"✓ Summary generated: {summarizer_result.num_bullets} bullet points, {summarizer_result.summary_length} characters")
        logger.info(f"Summary preview: {summarizer_result.summary[:200]}...")
        
        # Step 3: Decision Agent
        logger.info("\n" + "="*80)
        logger.info("STEP 3: DECISION AGENT")
        logger.info("="*80)
        decision_result: DecisionResult = run_decision(
            summary=summarizer_result.summary,
            model_name=decision_model,
            temperature=temperature
        )
        
        if not decision_result.success:
            logger.error(f"Decision agent failed: {decision_result.error_message}")
            return False
        
        logger.info(f"✓ Decision generated: Action required={decision_result.action_required}, {len(decision_result.next_steps)} next steps")
        logger.info(f"Decision preview: {decision_result.decision[:200]}...")
        
        # Step 4: Reporter Agent
        logger.info("\n" + "="*80)
        logger.info("STEP 4: REPORTER AGENT")
        logger.info("="*80)
        reporter_result: ReporterResult = run_reporter(
            summary=summarizer_result.summary,
            decision=decision_result.decision,
            model_name=reporter_model,
            temperature=temperature,
            max_words=150
        )
        
        if not reporter_result.success:
            logger.error(f"Reporter agent failed: {reporter_result.error_message}")
            return False
        
        logger.info(f"✓ Report generated: {reporter_result.word_count} words, {reporter_result.report_length} characters")
        logger.info(f"Report preview: {reporter_result.report[:200]}...")
        
        # Step 5: Report Generation Agent
        logger.info("\n" + "="*80)
        logger.info("STEP 5: REPORT GENERATION AGENT")
        logger.info("="*80)
        report_gen_result: ReportGenerationResult = run_report_generation(
            summary=summarizer_result.summary,
            decision=decision_result.decision,
            reporter_output=reporter_result.report,
            ingestion_metadata=ingestion_result.metadata,
            output_dir=output_dir,
            output_filename=report_filename or REPORT_FILENAME,
            json_filename=json_filename or JSON_REPORT_FILENAME
        )
        
        if not report_gen_result.success:
            logger.error(f"Report generation failed: {report_gen_result.error_message}")
            return False
        
        logger.info(f"✓ Final report generated: {report_gen_result.report_path}")
        logger.info(f"✓ JSON report generated: {report_gen_result.json_path}")
        
        # Workflow completion summary
        logger.info("\n" + "="*80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Documents processed: {ingestion_result.metadata['num_documents']}")
        logger.info(f"Summary: {summarizer_result.num_bullets} bullet points")
        logger.info(f"Decision: Action required = {decision_result.action_required}")
        logger.info(f"Report: {reporter_result.word_count} words")
        logger.info(f"Output files:")
        logger.info(f"  - {report_gen_result.report_path}")
        logger.info(f"  - {report_gen_result.json_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("WORKFLOW EXECUTION SUMMARY")
        print("="*80)
        print(f"✓ Document Ingestion: {ingestion_result.metadata['num_documents']} documents")
        print(f"✓ Summary: {summarizer_result.num_bullets} bullet points")
        print(f"✓ Decision: Action required = {decision_result.action_required}")
        print(f"✓ Report: {reporter_result.word_count} words")
        print(f"\nReports saved to:")
        print(f"  - {report_gen_result.report_path}")
        print(f"  - {report_gen_result.json_path}")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.exception(f"Fatal error in workflow execution: {e}")
        return False


def main():
    """Main entry point for the workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the agentic workflow pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Directory containing input text files (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory for output reports (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--summarizer-model",
        type=str,
        default=None,
        help=f"Model for summarizer agent (default: {SUMMARIZER_MODEL})"
    )
    parser.add_argument(
        "--decision-model",
        type=str,
        default=None,
        help=f"Model for decision agent (default: {DECISION_MODEL})"
    )
    parser.add_argument(
        "--reporter-model",
        type=str,
        default=None,
        help=f"Model for reporter agent (default: {REPORTER_MODEL})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Temperature for all agents (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--report-filename",
        type=str,
        default=REPORT_FILENAME,
        help=f"Name of the text report file (default: {REPORT_FILENAME})"
    )
    parser.add_argument(
        "--json-report-filename",
        type=str,
        default=JSON_REPORT_FILENAME,
        help=f"Name of the JSON report file (default: {JSON_REPORT_FILENAME})"
    )
    
    args = parser.parse_args()
    
    # Run workflow with timing
    success, duration = run_workflow_with_timing(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        summarizer_model=args.summarizer_model,
        decision_model=args.decision_model,
        reporter_model=args.reporter_model,
        temperature=args.temperature,
        report_filename=args.report_filename,
        json_filename=args.json_report_filename
    )
    
    if not success:
        logger.error("Workflow execution failed")
        exit(1)


if __name__ == "__main__":
    main()
