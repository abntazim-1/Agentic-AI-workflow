"""
Pipeline Runner Script
Simple interface to run the agentic workflow pipeline.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import run_workflow_with_timing, DATA_DIR, OUTPUT_DIR
from logging_config import get_logger

logger = get_logger("run_pipeline")


def print_banner():
    """Print a welcome banner."""
    banner = """
    ====================================================================
            AGENTIC WORKFLOW PIPELINE RUNNER
    ====================================================================
    Document Ingestion -> Summarize -> Decide -> Report -> Output
    ====================================================================
    """
    print(banner)


def print_usage():
    """Print usage information."""
    usage = """
    Usage:
        python run_pipeline.py [options]
    
    Options:
        --data-dir DIR          Directory containing input text files
                                (default: artifacts/)
        
        --output-dir DIR        Directory for output reports
                                (default: current directory)
        
        --model MODEL           Use same model for all agents
                                (default: tinyllama:1.1b)
        
        --summarizer MODEL      Model for summarizer agent
        --decision MODEL        Model for decision agent
        --reporter MODEL        Model for reporter agent
        
        --temperature FLOAT     Temperature setting (0.0-2.0)
                                (default: 0.2)
        
        --report-name NAME      Name of the text report file
                                (default: report.txt)
        
        --json-name NAME        Name of the JSON report file
                                (default: report.json)
        
        --help                  Show this help message
    
    Examples:
        # Run with defaults
        python run_pipeline.py
        
        # Use specific data directory
        python run_pipeline.py --data-dir ./data
        
        # Use different models for each agent
        python run_pipeline.py --summarizer llama3:8b --decision mistral:7b
        
        # Custom output location
        python run_pipeline.py --output-dir ./reports --report-name my_report.txt
        
        # Higher creativity (temperature)
        python run_pipeline.py --temperature 0.7
    """
    print(usage)


def parse_args():
    """Parse command-line arguments."""
    args = {
        'data_dir': None,
        'output_dir': None,
        'summarizer_model': None,
        'decision_model': None,
        'reporter_model': None,
        'temperature': 0.2,
        'report_filename': 'report.txt',
        'json_filename': 'report.json'
    }
    
    # Simple argument parser
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--help' or arg == '-h':
            print_usage()
            sys.exit(0)
        
        elif arg == '--data-dir' and i + 1 < len(sys.argv):
            args['data_dir'] = Path(sys.argv[i + 1])
            i += 2
        
        elif arg == '--output-dir' and i + 1 < len(sys.argv):
            args['output_dir'] = Path(sys.argv[i + 1])
            i += 2
        
        elif arg == '--model' and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            args['summarizer_model'] = model
            args['decision_model'] = model
            args['reporter_model'] = model
            i += 2
        
        elif arg == '--summarizer' and i + 1 < len(sys.argv):
            args['summarizer_model'] = sys.argv[i + 1]
            i += 2
        
        elif arg == '--decision' and i + 1 < len(sys.argv):
            args['decision_model'] = sys.argv[i + 1]
            i += 2
        
        elif arg == '--reporter' and i + 1 < len(sys.argv):
            args['reporter_model'] = sys.argv[i + 1]
            i += 2
        
        elif arg == '--temperature' and i + 1 < len(sys.argv):
            try:
                args['temperature'] = float(sys.argv[i + 1])
            except ValueError:
                print(f"Error: Invalid temperature value: {sys.argv[i + 1]}")
                sys.exit(1)
            i += 2
        
        elif arg == '--report-name' and i + 1 < len(sys.argv):
            args['report_filename'] = sys.argv[i + 1]
            i += 2
        
        elif arg == '--json-name' and i + 1 < len(sys.argv):
            args['json_filename'] = sys.argv[i + 1]
            i += 2
        
        else:
            print(f"Error: Unknown argument: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    
    return args


def validate_args(args):
    """Validate command-line arguments."""
    errors = []
    
    # Validate data directory
    if args['data_dir']:
        if not args['data_dir'].exists():
            errors.append(f"Data directory does not exist: {args['data_dir']}")
        elif not args['data_dir'].is_dir():
            errors.append(f"Path is not a directory: {args['data_dir']}")
    
    # Validate temperature
    if not (0.0 <= args['temperature'] <= 2.0):
        errors.append(f"Temperature must be between 0.0 and 2.0, got: {args['temperature']}")
    
    # Validate output directory (create if doesn't exist)
    if args['output_dir']:
        try:
            args['output_dir'].mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
    
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def run():
    """Main pipeline runner function."""
    print_banner()
    
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*70)
    print("PIPELINE CONFIGURATION")
    print("="*70)
    print(f"Data directory:      {args['data_dir'] or DATA_DIR}")
    print(f"Output directory:    {args['output_dir'] or OUTPUT_DIR}")
    print(f"Summarizer model:    {args['summarizer_model'] or 'default'}")
    print(f"Decision model:      {args['decision_model'] or 'default'}")
    print(f"Reporter model:      {args['reporter_model'] or 'default'}")
    print(f"Temperature:         {args['temperature']}")
    print(f"Report filename:     {args['report_filename']}")
    print(f"JSON filename:       {args['json_filename']}")
    print("="*70 + "\n")
    
    # Run the workflow with timing
    try:
        success, duration = run_workflow_with_timing(
            data_dir=args['data_dir'],
            output_dir=args['output_dir'],
            summarizer_model=args['summarizer_model'],
            decision_model=args['decision_model'],
            reporter_model=args['reporter_model'],
            temperature=args['temperature'],
            report_filename=args['report_filename'],
            json_filename=args['json_filename']
        )
        
        if success:
            print("\n" + "="*70)
            print("[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("="*70)
            sys.exit(0)
        else:
            print("\n" + "="*70)
            print("[FAILED] PIPELINE EXECUTION FAILED")
            print("="*70)
            print("Check the logs for detailed error information.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nPipeline execution interrupted by user.")
        sys.exit(130)
    
    except Exception as e:
        logger.exception(f"Fatal error in pipeline runner: {e}")
        print(f"\nFatal error: {e}")
        print("Check the logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    run()

