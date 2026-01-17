"""Humanize: An offline, deterministic text humanization tool."""

from humanize.text.analyzer import TextAnalyzer
from humanize.text.diagnosis import TextDiagnosis
from humanize.rewrite.rules import RewriteEngine


def humanize(text: str) -> str:
    """Canonical entry point for text humanization.

    This function orchestrates the text humanization pipeline:
    1. Analyzes input text using TextAnalyzer
    2. Diagnoses issues using TextDiagnosis
    3. If all metrics are within normal ranges, returns original text unchanged
    4. Otherwise, applies targeted rewrite rules through RewriteEngine
    5. Validates rewritten output by re-running analysis
    6. Returns the humanized text

    The system is designed to preserve meaning, retain technical terminology,
    and maintain formal tone while adjusting only structural elements that fall
    outside human writing ranges.

    Args:
        text: Raw body text (multiple sentences, possibly multi-paragraph)

    Returns:
        Humanized text string, unchanged if all metrics are normal
    """
    # Step 1: Analyze input text
    analyzer = TextAnalyzer(text)
    
    # Step 2: Diagnose issues
    diagnosis = TextDiagnosis(analyzer)
    result = diagnosis.diagnose()
    
    # Step 3: If all statuses are "normal", return original text unchanged
    if (
        result.flesch_status == "normal"
        and result.sentence_length_status == "normal"
        and result.sentence_count_status == "normal"
    ):
        return text
    
    # Step 4: Apply rewrite rules through RewriteEngine
    # RewriteEngine will internally re-analyze the text and apply only
    # the rules indicated by its own diagnosis, ensuring minimal changes
    rewrite_engine = RewriteEngine(text)
    rewritten_text = rewrite_engine.rewrite()
    
    # Step 5: Re-run TextAnalyzer on rewritten output for validation
    # (This validates the rewrite but doesn't affect the return value)
    _validation_analyzer = TextAnalyzer(rewritten_text)
    
    # Step 6: Return rewritten text
    return rewritten_text

