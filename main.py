
# main.py
"""
Local Agentic AI Workflow Simulation (3 agents)
- Read -> Summarize -> Decide -> Report
Requires: Ollama (daemon running + model pulled), LangChain, langchain-ollama
"""

import os
from pathlib import Path
from typing import List, Dict

# Logging
from logging_config import get_logger

# LangChain & Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

logger = get_logger("main")

# -------- CONFIG --------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # or "gemma:2b", "llama3", etc.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DATA_DIR = Path("data/")

# --------- helpers ----------
def read_text_files(folder: Path) -> List[Dict]:
    """Read all .txt files into a list of dicts {filename, text}"""
    docs = []
    folder = Path(folder).resolve()  # Ensure absolute path
    
    if not folder.exists():
        logger.error(f"Data directory does not exist: {folder}")
        return docs
    
    logger.info(f"Reading files from: {folder}")
    
    # Get all .txt files
    txt_files = list(folder.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} .txt files: {[f.name for f in txt_files]}")
    
    for file_path in txt_files:
        try:
            # Read with error handling and encoding fallback
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {file_path.name}, trying latin-1")
                text = file_path.read_text(encoding="latin-1")
            
            # Strip whitespace but keep structure
            text = text.strip()
            
            if not text:
                logger.warning(f"File {file_path.name} is empty, skipping")
                continue
            
            docs.append({
                "filename": file_path.name,
                "text": text,
                "filepath": str(file_path)
            })
            logger.info(f"Successfully read {file_path.name} ({len(text)} characters)")
            
        except Exception as e:
            logger.error(f"Error reading file {file_path.name}: {e}")
            continue
    
    return docs

def combine_documents(docs: List[Dict]) -> str:
    """Combine all documents into a single text string with proper formatting"""
    if not docs:
        logger.warning("No documents to combine")
        return ""
    
    combined_parts = []
    for d in docs:
        filename = d.get('filename', 'unknown')
        text = d.get('text', '')
        
        # Ensure text is not empty
        if not text or not text.strip():
            logger.warning(f"Skipping empty document: {filename}")
            continue
        
        # Format: filename header followed by content
        combined_parts.append(f"=== {filename} ===\n{text.strip()}\n")
    
    if not combined_parts:
        logger.error("No valid document content to combine")
        return ""
    
    combined_text = "\n".join(combined_parts).strip()
    logger.info(f"Combined {len(docs)} documents into {len(combined_text)} characters")
    
    return combined_text

# --------- Summarizer Agent ----------
def summarizer_agent(llm, document_text: str):
    """
    Summarizes the document text into concise bullet points.
    """
    # Validate input
    if not document_text or not document_text.strip():
        logger.error("Empty document text provided to summarizer agent")
        return "Error: No content to summarize"
    
    logger.info(f"Summarizing document with {len(document_text)} characters")
    logger.debug(f"Document preview: {document_text[:200]}...")
    
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in 3 concise bullet points:\n\n{document}\n\nBullets:"
    )
    chain = prompt | llm
    
    try:
        output = chain.invoke({"document": document_text}).content
        logger.info(f"Summary generated successfully ({len(output)} characters)")
        return output
    except Exception as e:
        logger.exception(f"Error in summarizer agent: {e}")
        raise

# --------- Decision Agent ----------
def decision_agent(llm, summary_text: str):
    """
    Based on summary, decide whether action is required and provide next steps.
    """
    prompt = ChatPromptTemplate.from_template(
        "Given this summary:\n{summary_text}\n\nDecide whether any action is required (Yes/No). "
        "Provide a short justification and two next steps."
    )
    chain = prompt | llm
    decision = chain.invoke({"summary_text": summary_text}).content
    return decision

# --------- Reporter Agent ----------
def reporter_agent(llm, summary_text: str, decision_text: str):
    """
    Produce a final human-facing report combining summary and decision.
    """
    prompt = ChatPromptTemplate.from_template(
        "Combine the following information into a concise professional report for a stakeholder. "
        "Limit to 150 words.\n\nSummary:\n{summary_text}\n\nDecision & Steps:\n{decision}"
    )
    chain = prompt | llm
    report_text = chain.invoke({
        "summary_text": summary_text,
        "decision": decision_text
    }).content
    return report_text

# --------- setup local Ollama client (LangChain wrapper) ----------
def make_ollama_client(model_name: str = None):
    """Create an Ollama LLM client. Optionally specify a different model for this task."""
    model = model_name or OLLAMA_MODEL
    client = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0.2)
    return client

# --------- main flow ----------
def run_pipeline():
    logger.info("=== Local Agentic Workflow Demo ===")
    docs = read_text_files(DATA_DIR)
    logger.info(f"Found {len(docs)} documents in {DATA_DIR}")
    
    if not docs:
        logger.warning("No documents found. Please add .txt files to the data/ directory.")
        return
    
    # Display the documents that were read
    print("\n" + "="*60)
    print("DOCUMENTS READ FROM DATA FOLDER:")
    print("="*60)
    for doc in docs:
        print(f"\n--- {doc['filename']} ---")
        print(doc['text'])
        print("-" * 60)
    print("\n")
    
    # Combine all documents into a single text
    combined_text = combine_documents(docs)
    
    if not combined_text or not combined_text.strip():
        logger.error("No content available to process. Check if documents contain text.")
        print("\nERROR: No document content available for processing!")
        return
    
    logger.info(f"Combined document length: {len(combined_text)} characters")
    
    # Debug: Show a preview of what will be sent to LLM
    print("\n" + "="*60)
    print("COMBINED TEXT TO BE SENT TO LLM (first 500 chars):")
    print("="*60)
    print(combined_text[:500])
    if len(combined_text) > 500:
        print(f"... ({len(combined_text) - 500} more characters)")
    print("="*60 + "\n")

    # Initialize LLM clients (can use different models for different tasks if needed)
    logger.info("Initializing Ollama LLM clients...")
    llm_summary = make_ollama_client()  # Can specify different model: make_ollama_client("mistral")
    llm_decision = make_ollama_client()  # Can specify different model: make_ollama_client("llama3")
    llm_reporter = make_ollama_client()  # Can specify different model: make_ollama_client("gemma")

    # 1) Summarize
    logger.info("Starting summarizer agent...")
    summary = summarizer_agent(llm_summary, combined_text)
    logger.info(f"Summary:\n{summary}")

    # 2) Decision making
    logger.info("Starting decision agent...")
    decision = decision_agent(llm_decision, summary)
    logger.info(f"Decision:\n{decision}")

    # 3) Reporting
    logger.info("Starting reporter agent...")
    report = reporter_agent(llm_reporter, summary, decision)
    logger.info(f"\n--- FINAL REPORT ---\n{report}")
    
    # Save report
    Path("report.txt").write_text(report, encoding="utf-8")
    logger.info("Report saved to report.txt")
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.exception("Fatal error in pipeline execution")
        raise
