"""
Simple utility to measure workflow execution time.

Usage:
    python time_workflow.py
    
    Or import and use:
    from time_workflow import measure_workflow_time
    success, duration = measure_workflow_time()
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import run_workflow_with_timing, DATA_DIR, OUTPUT_DIR
from logging_config import get_logger

logger = get_logger("time_workflow")


def measure_workflow_time(
    data_dir=None,
    output_dir=None,
    summarizer_model=None,
    decision_model=None,
    reporter_model=None,
    temperature=0.2,
    report_filename=None,
    json_filename=None
):
    """
    Simple function to measure workflow execution time.
    
    Args:
        data_dir: Directory containing input text files (default: DATA_DIR)
        output_dir: Directory for output reports (default: OUTPUT_DIR)
        summarizer_model: Model name for summarizer agent
        decision_model: Model name for decision agent
        reporter_model: Model name for reporter agent
        temperature: Temperature setting for all agents (default: 0.2)
        report_filename: Name of the text report file
        json_filename: Name of the JSON report file
        
    Returns:
        Tuple of (success: bool, duration: float) where duration is in seconds
    """
    print("="*80)
    print("MEASURING WORKFLOW EXECUTION TIME")
    print("="*80)
    
    success, duration = run_workflow_with_timing(
        data_dir=data_dir,
        output_dir=output_dir,
        summarizer_model=summarizer_model,
        decision_model=decision_model,
        reporter_model=reporter_model,
        temperature=temperature,
        report_filename=report_filename,
        json_filename=json_filename
    )
    
    return success, duration


def main():
    """Main entry point for the timing script."""
    success, duration = measure_workflow_time()
    
    if success:
        print(f"\n✓ Workflow completed successfully in {duration:.2f} seconds")
        return 0
    else:
        print(f"\n✗ Workflow failed after {duration:.2f} seconds")
        return 1


if __name__ == "__main__":
    sys.exit(main())

