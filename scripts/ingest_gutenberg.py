#!/usr/bin/env python3
"""
Standalone script to ingest Project Gutenberg text files into the writing_events dataset.

Reads .txt files from datasets/books/, removes boilerplate, filters paragraphs,
and inserts writing events using DatasetManager.bulk_append.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict

# Import DatasetManager from src.data.arrow
import sys

# Add src to path for import
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data.arrow import DatasetManager


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def remove_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate (header and footer).
    
    Finds the START marker and extracts text until END marker (or end of file).
    Also removes page markers like [Pg 1], [Pg i].
    """
    # Find START marker (case-insensitive)
    start_pattern = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
    start_match = start_pattern.search(text)
    
    if not start_match:
        logger.warning("No START marker found, processing entire file")
        start_pos = 0
    else:
        start_pos = start_match.end()
    
    # Find END marker (case-insensitive)
    end_pattern = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
    end_match = end_pattern.search(text)
    
    if end_match:
        end_pos = end_match.start()
        content = text[start_pos:end_pos]
    else:
        logger.warning("No END marker found, using text until end of file")
        content = text[start_pos:]
    
    # Remove page markers like [Pg 1], [Pg i], [Pg iv]
    page_marker_pattern = re.compile(r"\[Pg [^\]]+\]")
    content = page_marker_pattern.sub("", content)
    
    return content.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs using blank lines (double newlines)."""
    # Split by double newlines or more
    paragraphs = re.split(r"\n\s*\n", text)
    
    # Strip whitespace and filter empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def count_sentences(text: str) -> int:
    """Count sentences by finding sentence-ending punctuation followed by space or end."""
    # Count sentence endings: . ! ? followed by space or end of string
    pattern = r"[.!?]+(?=\s|$)"
    matches = re.findall(pattern, text)
    return len(matches) if matches else 0


def contains_punctuation(text: str) -> bool:
    """Check if text contains sentence-ending punctuation."""
    return bool(re.search(r"[.!?]", text))


def is_list(text: str) -> bool:
    """
    Check if paragraph is a list (bullets, numbered items, or multiple list-like lines).
    """
    lines = text.split("\n")
    
    # Check if starts with bullet markers
    first_line = lines[0].strip()
    if re.match(r"^[-*•]\s", first_line):
        return True
    
    # Check if starts with numbered items
    if re.match(r"^\d+[.)]\s", first_line):
        return True
    
    # Check if multiple lines each start with list markers
    list_line_count = 0
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[-*•]\s", stripped) or re.match(r"^\d+[.)]\s", stripped):
            list_line_count += 1
    
    # If more than 50% of lines are list items, consider it a list
    if len(lines) > 1 and list_line_count > len(lines) / 2:
        return True
    
    return False


def is_dialogue_heavy(text: str) -> bool:
    """
    Check if paragraph is dialogue-heavy (too many quotes).
    Reject if more than 60% of characters are within quotes or every line starts with quotes.
    """
    lines = text.split("\n")
    
    # Check if every line starts with a quote character
    if len(lines) > 1:
        quote_start_count = sum(1 for line in lines if line.strip().startswith('"') or line.strip().startswith("'"))
        if quote_start_count == len(lines):
            return True
    
    # Count characters within quotes
    total_chars = len(text.replace("\n", "").replace(" ", ""))
    if total_chars == 0:
        return False
    
    # Find all quoted sections (handling both single and double quotes)
    quoted_pattern = re.compile(r'["\'].*?["\']')
    quoted_text = "".join(quoted_pattern.findall(text))
    quoted_chars = len(quoted_text.replace("\n", "").replace(" ", ""))
    
    # If more than 60% of non-whitespace characters are in quotes, it's dialogue-heavy
    if quoted_chars > total_chars * 0.6:
        return True
    
    return False


def is_table(text: str) -> bool:
    """Check if paragraph looks like a table (multiple pipes or tab-separated columns)."""
    # Check for multiple pipe characters (markdown-style tables)
    if text.count("|") > 2:
        return True
    
    # Check for tab-separated columns (multiple tabs on same line)
    lines = text.split("\n")
    for line in lines:
        if line.count("\t") > 2:
            return True
    
    return False


def filter_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    Filter paragraphs according to requirements:
    - 1-5 sentences
    - Contains punctuation
    - Not a list, table, or dialogue-heavy
    """
    filtered = []
    
    for para in paragraphs:
        # Check sentence count
        sentence_count = count_sentences(para)
        if sentence_count < 1 or sentence_count > 5:
            continue
        
        # Check punctuation
        if not contains_punctuation(para):
            continue
        
        # Check if it's a list
        if is_list(para):
            continue
        
        # Check if it's dialogue-heavy
        if is_dialogue_heavy(para):
            continue
        
        # Check if it's a table
        if is_table(para):
            continue
        
        filtered.append(para)
    
    return filtered


def create_records(paragraphs: List[str]) -> List[Dict[str, str]]:
    """Create records for bulk insertion."""
    records = []
    for para in paragraphs:
        records.append({
            "text": para,
            "source": "gutenberg",
            "license": "public_domain"
        })
    return records


def batch_insert(dataset_manager: DatasetManager, records: List[Dict[str, str]], batch_size: int = 500) -> int:
    """Insert records in batches using bulk_append."""
    total_inserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            dataset_manager.bulk_append(batch)
            total_inserted += len(batch)
            logger.info(f"Inserted batch: {len(batch)} records (total: {total_inserted})")
        except Exception as e:
            logger.error(f"Error inserting batch {i // batch_size + 1}: {e}")
            raise
    
    return total_inserted


def process_file(filepath: Path, dataset_manager: DatasetManager) -> tuple[int, int, int]:
    """
    Process a single Gutenberg text file.
    
    Returns: (paragraphs_extracted, paragraphs_after_filtering, records_inserted)
    """
    logger.info(f"Processing: {filepath.name}")
    
    try:
        # Read file with UTF-8 encoding, handle errors gracefully
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Error reading {filepath.name}: {e}")
        return (0, 0, 0)
    
    # Remove boilerplate
    cleaned_text = remove_gutenberg_boilerplate(content)
    
    if not cleaned_text:
        logger.warning(f"No content after removing boilerplate: {filepath.name}")
        return (0, 0, 0)
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(cleaned_text)
    paragraphs_extracted = len(paragraphs)
    
    logger.info(f"  Extracted {paragraphs_extracted} paragraphs")
    
    # Filter paragraphs
    filtered_paragraphs = filter_paragraphs(paragraphs)
    paragraphs_after_filtering = len(filtered_paragraphs)
    
    logger.info(f"  After filtering: {paragraphs_after_filtering} paragraphs")
    
    if not filtered_paragraphs:
        logger.warning(f"No valid paragraphs after filtering: {filepath.name}")
        return (paragraphs_extracted, 0, 0)
    
    # Create records
    records = create_records(filtered_paragraphs)
    
    # Batch insert
    records_inserted = batch_insert(dataset_manager, records)
    
    return (paragraphs_extracted, paragraphs_after_filtering, records_inserted)


def main():
    """Main execution function."""
    # Initialize DatasetManager
    dataset_path = Path("datasets/data.parquet")
    dataset_manager = DatasetManager(str(dataset_path))
    logger.info(f"Initialized DatasetManager: {dataset_path}")
    
    # Find all .txt files in datasets/books/
    books_dir = Path("datasets/books")
    if not books_dir.exists():
        logger.error(f"Directory not found: {books_dir}")
        return
    
    txt_files = sorted(books_dir.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"No .txt files found in {books_dir}")
        return
    
    logger.info(f"Found {len(txt_files)} file(s) to process")
    
    # Process each file
    total_extracted = 0
    total_filtered = 0
    total_inserted = 0
    
    for txt_file in txt_files:
        extracted, filtered, inserted = process_file(txt_file, dataset_manager)
        total_extracted += extracted
        total_filtered += filtered
        total_inserted += inserted
        logger.info("")
    
    # Summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"  Total paragraphs extracted: {total_extracted}")
    logger.info(f"  Total paragraphs after filtering: {total_filtered}")
    logger.info(f"  Total records inserted: {total_inserted}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

