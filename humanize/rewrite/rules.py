"""Deterministic rule-based rewrite engine for structural text humanization.

This module implements targeted rewrite rules that adjust sentence structure
based on text diagnosis, without altering meaning or introducing new information.
"""

import re
from typing import List

from humanize.text.analyzer import TextAnalyzer
from humanize.text.diagnosis import TextDiagnosis


def _count_words(sentence: str) -> int:
    """Count words in a sentence, stripping punctuation.

    Args:
        sentence: The sentence to count words in

    Returns:
        Number of words in the sentence
    """
    words = re.split(r"\s+", sentence.strip())
    count = 0
    for word in words:
        cleaned = word.strip(".,!?;:()[]{}'\"-—_")
        if cleaned:
            count += 1
    return count


def _is_decimal_number_or_acronym(text: str, pos: int) -> bool:
    """Check if a comma at position pos is part of a decimal number or acronym.

    Args:
        text: The text to check
        pos: Position of the comma

    Returns:
        True if comma is part of number/acronym and should not be split on
    """
    if pos < 0 or pos >= len(text) or text[pos] != ",":
        return False

    # Check for decimal numbers: digit, comma, digit (e.g., 50,000 or 1,234.56)
    before_context = text[max(0, pos - 10):pos]
    after_context = text[pos + 1:min(len(text), pos + 11)]
    
    # Decimal number pattern: digits before comma, digits after
    if re.search(r'\d+$', before_context) and re.search(r'^\s*\d+', after_context):
        return True
    
    # Check for acronyms: capital letters separated by comma-space
    # Pattern: 2+ capital letters, comma, space, 2+ capital letters
    # e.g., "HFT, DLT" or "NASA, ESA"
    acronym_pattern = r'\b[A-Z]{2,}\s*,\s*[A-Z]{2,}\b'
    context_text = text[max(0, pos - 10):min(len(text), pos + 11)]
    match = re.search(acronym_pattern, context_text, re.IGNORECASE)
    if match:
        # Verify the match contains the comma at our position
        match_start = max(0, pos - 10) + match.start()
        match_end = match_start + len(match.group())
        if match_start <= pos < match_end:
            return True
    
    # Check for common abbreviations with comma (e.g., "e.g.," "i.e.,")
    abbrev_pattern = r'\b(e\.g\.|i\.e\.|etc\.|vs\.|cf\.)\s*,'
    if re.search(abbrev_pattern, text[max(0, pos - 8):pos + 1], re.IGNORECASE):
        return True
    
    return False


def _is_academic_intro_clause(text: str, split_pos: int) -> bool:
    """Check if splitting at split_pos would break an academic intro clause.

    Academic intro clauses are phrases like "By analyzing...", "In this paper...",
    "The objective of...", "Through examining..." that should not be separated
    from their main clause unless the sentence is extremely dense.

    Args:
        text: The sentence text
        split_pos: Proposed split position

    Returns:
        True if splitting here would incorrectly break an academic intro clause
    """
    # Look before split point for academic intro patterns
    before_text = text[:split_pos].strip()
    
    # Academic intro patterns (case-insensitive)
    intro_patterns = [
        r'\bby\s+(analyzing|examining|investigating|studying|exploring|reviewing)\b',
        r'\bin\s+(this|the)\s+(paper|study|research|article|work|analysis)\b',
        r'\bthe\s+(objective|purpose|aim|goal)\s+of\s+(this|the)\b',
        r'\bthrough\s+(analyzing|examining|investigating|studying)\b',
        r'\bthis\s+(paper|study|research|article)\s+(argues|demonstrates|shows|proposes)\b',
    ]
    
    for pattern in intro_patterns:
        matches = list(re.finditer(pattern, before_text, re.IGNORECASE))
        if matches:
            # If we found an intro pattern before the split, this might be an intro clause
            # Only allow split if sentence is extremely dense (>40 words)
            word_count = _count_words(text)
            if word_count <= 40:
                return True  # Don't split intro clauses unless extremely dense
    
    return False


def _normalize_sentence_fragment(fragment: str, is_sentence_start: bool = True) -> str:
    """Normalize a sentence fragment with proper capitalization and punctuation.

    Args:
        fragment: The fragment to normalize
        is_sentence_start: True if this is a sentence start (should be capitalized)

    Returns:
        Normalized fragment with proper spacing, capitalization, and punctuation
    """
    fragment = fragment.strip()
    
    if not fragment:
        return fragment
    
    # Only capitalize if it's actually a sentence start
    # Don't capitalize mid-sentence fragments (preserve lowercase after comma/conjunction)
    if is_sentence_start and fragment and fragment[0].islower():
        fragment = fragment[0].upper() + fragment[1:]
    
    return fragment


def _merge_sentences(s1: str, s2: str, connector: str = ", and") -> str:
    """Merge two sentences with a connector.

    Handles punctuation and capitalization safely to avoid run-on sentences.
    Preserves proper capitalization of second sentence if it's a proper noun
    or intentional capitalization.

    Args:
        s1: First sentence
        s2: Second sentence
        connector: Connector to use between sentences (default: ", and")

    Returns:
        Merged sentence with proper punctuation and spacing
    """
    # Clean first sentence: remove trailing sentence-ending punctuation
    s1_clean = s1.rstrip(".,!?;:").rstrip()
    
    # Clean second sentence: remove leading punctuation but preserve content
    s2_clean = s2.lstrip(".,!?;:").lstrip()
    
    # If second sentence starts with capital letter and it's not a proper noun
    # (i.e., it's likely a new sentence start), lowercase it for merging
    if s2_clean and s2_clean[0].isupper():
        # Only lowercase if it's likely not a proper noun
        # Heuristic: if first word is common article/determiner, it's likely not proper
        first_word = s2_clean.split()[0] if s2_clean.split() else ""
        common_words = {"The", "A", "An", "This", "That", "These", "Those", "Some", "All", "Each", "Every"}
        if first_word in common_words:
            s2_clean = s2_clean[0].lower() + s2_clean[1:]
    
    # Ensure connector has proper spacing
    if not connector.startswith(" "):
        connector = " " + connector
    
    # Merge with connector
    merged = f"{s1_clean}{connector} {s2_clean}"
    
    # Validate: check if merged sentence isn't too long (avoid run-ons)
    word_count = _count_words(merged)
    if word_count > 50:  # Reasonable limit to avoid run-ons
        # If too long, keep original format (though this shouldn't happen with normal merging)
        pass
    
    return merged


def merge_short_sentences(sentences: List[str], target_avg: float = 12.0) -> List[str]:
    """Rule 1: Merge adjacent short sentences when text is too choppy.

    Merges pairs of short sentences using simple connectors (comma, ", and").
    Prefers merging pairs and does not exceed target average length.

    Args:
        sentences: List of sentence strings
        target_avg: Target average words per sentence after merging

    Returns:
        List of sentences with short ones merged
    """
    if not sentences:
        return sentences

    result = []
    i = 0

    while i < len(sentences):
        current = sentences[i]
        current_words = _count_words(current)

        # If current sentence is already long enough, keep it as-is
        if current_words >= target_avg or i == len(sentences) - 1:
            result.append(current)
            i += 1
            continue

        # Look ahead to next sentence
        next_idx = i + 1
        if next_idx < len(sentences):
            next_sent = sentences[next_idx]
            next_words = _count_words(next_sent)

            # Calculate average if we merge
            combined_words = current_words + next_words
            avg_after_merge = combined_words / (len(result) + 1)

            # Merge if both are short and average stays reasonable
            if current_words < target_avg and next_words < target_avg:
                if avg_after_merge <= target_avg * 1.5:  # Don't exceed target too much
                    merged = _merge_sentences(current, next_sent)
                    result.append(merged)
                    i += 2  # Skip both sentences
                    continue

        # Keep current sentence without merging
        result.append(current)
        i += 1

    return result


def _find_split_points(sentence: str, protect_academic: bool = True) -> List[int]:
    """Find potential split points in a sentence (commas, conjunctions).

    Excludes split points that would break:
    - Decimal numbers (e.g., 50,000)
    - Acronyms (e.g., HFT, DLT)
    - Academic intro clauses (unless sentence is extremely dense)

    Args:
        sentence: The sentence to analyze
        protect_academic: If True, protect academic intro clauses from splitting

    Returns:
        List of character positions where sentence could be split
    """
    points = []
    word_count = _count_words(sentence)
    # Only allow splitting intro clauses if extremely dense
    allow_academic_split = word_count > 40
    
    # Look for commas followed by spaces (but not inside parentheses)
    paren_depth = 0
    for i, char in enumerate(sentence):
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "," and paren_depth == 0:
            # Check if there's a space after and not at sentence end
            if i + 1 < len(sentence) and sentence[i + 1] == " ":
                # Don't split right after the start
                if i > 5:
                    # Skip if comma is part of decimal number or acronym
                    if _is_decimal_number_or_acronym(sentence, i):
                        continue
                    
                    # Skip if this would break academic intro clause (unless very dense)
                    if protect_academic and not allow_academic_split:
                        if _is_academic_intro_clause(sentence, i + 1):
                            continue
                    
                    points.append(i + 1)  # Position after comma and space

    # Look for coordinating conjunctions (and, but, which, that)
    # Pattern matches word boundaries to avoid false positives
    conj_patterns = [
        r"\b and ",
        r"\b but ",
        r"\b which ",
        r"\b that ",
    ]
    for pattern in conj_patterns:
        for match in re.finditer(pattern, sentence, re.IGNORECASE):
            # Only consider if not inside parentheses
            before_match = sentence[: match.start()]
            paren_depth = before_match.count("(") - before_match.count(")")
            if paren_depth == 0 and match.start() > 5:
                # Skip if this would break academic intro clause (unless very dense)
                if protect_academic and not allow_academic_split:
                    if _is_academic_intro_clause(sentence, match.end()):
                        continue
                
                points.append(match.end())

    # Sort and deduplicate
    points = sorted(set(points))
    return points


def split_long_sentences(sentences: List[str], min_words: int = 5) -> List[str]:
    """Rule 2: Split long sentences at commas and coordinating conjunctions.

    Splits sentences that are too long at natural break points (commas,
    conjunctions like "and", "but", "which", "that"), ensuring resulting
    sentences have at least min_words words. Never splits decimal numbers,
    acronyms, or academic intro clauses unless extremely dense.

    Args:
        sentences: List of sentence strings
        min_words: Minimum words per resulting sentence

    Returns:
        List of sentences with long ones split
    """
    result = []

    for sentence in sentences:
        word_count = _count_words(sentence)

        # Only split if sentence is long (more than 30 words)
        if word_count <= 30:
            result.append(sentence)
            continue

        # Find potential split points (with academic prose protection)
        split_points = _find_split_points(sentence, protect_academic=True)

        if not split_points:
            # No good split points found, keep as-is
            result.append(sentence)
            continue

        # Try to split at points that create balanced sentences
        current_start = 0
        splits_made = False
        for split_pos in split_points:
            # Extract segment before split point
            segment = sentence[current_start:split_pos].strip()
            seg_words = _count_words(segment)

            # Only split if segment meets minimum word count
            if seg_words >= min_words:
                # Normalize segment: remove trailing punctuation, add period, capitalize
                segment = segment.rstrip(".,!?;:")
                if not segment.endswith("."):
                    segment += "."
                # Normalize as sentence start (capitalize)
                segment = _normalize_sentence_fragment(segment, is_sentence_start=True)
                result.append(segment)
                current_start = split_pos
                splits_made = True

        # Add remaining portion
        if current_start < len(sentence):
            remainder = sentence[current_start:].strip()
            if remainder:
                remainder = remainder.rstrip(".,!?;:")
                if not remainder.endswith("."):
                    remainder += "."
                # Only add if it meets minimum
                if _count_words(remainder) >= min_words:
                    # Normalize as sentence start (capitalize)
                    remainder = _normalize_sentence_fragment(remainder, is_sentence_start=True)
                    result.append(remainder)
                else:
                    # Merge with previous sentence if too short (don't capitalize)
                    if result:
                        remainder = remainder.rstrip(".,!?;:")
                        result[-1] = result[-1].rstrip(".,!?;:") + ", " + remainder
                    else:
                        # If no previous, still normalize as sentence start
                        remainder = _normalize_sentence_fragment(remainder, is_sentence_start=True)
                        result.append(remainder)

        # If no splits were made, keep original
        if not splits_made:
            result.append(sentence)

    return result


def simplify_parentheses(sentences: List[str]) -> List[str]:
    """Rule 3: Extract parenthetical clauses into standalone sentences when possible.

    Extracts parenthetical clauses from sentences when they can stand alone
    without losing meaning. Avoids duplication of content.

    Args:
        sentences: List of sentence strings

    Returns:
        List of sentences with parenthetical clauses extracted where appropriate
    """
    result = []

    for sentence in sentences:
        # Find all parenthetical expressions
        paren_pattern = r"\([^)]+\)"
        matches = list(re.finditer(paren_pattern, sentence))

        if not matches:
            result.append(sentence)
            continue

        # Check if we can extract the first substantial parenthetical
        # Only extract if it's a complete thought (starts with capital or is substantial)
        extracted = []
        modified_sentence = sentence

        for match in matches:
            paren_content = match.group(0)[1:-1]  # Remove parentheses
            paren_words = _count_words(paren_content)

            # Extract if substantial (at least 5 words) and seems standalone
            if paren_words >= 5:
                # Check if it starts with capital letter (likely a complete thought)
                if paren_content and paren_content[0].isupper():
                    # Extract this parenthetical
                    extracted.append(paren_content)
                    # Remove from original sentence
                    before = modified_sentence[: match.start()]
                    after = modified_sentence[match.end() :]
                    # Clean up spacing around removal point
                    before = before.rstrip()
                    after = after.lstrip()
                    # If there was a comma before, remove it
                    if before.endswith(","):
                        before = before[:-1].rstrip()
                    # Reconstruct sentence
                    modified_sentence = (before + " " + after).strip()
                    # Only extract one per sentence to avoid over-processing
                    break

        # If extraction occurred, add extracted parentheticals and modified sentence
        if extracted:
            # Capitalize extracted parentheticals and add to result
            for ext in extracted:
                ext = ext.strip()
                if ext and not ext.endswith("."):
                    ext += "."
                # Capitalize first letter
                if ext:
                    ext = ext[0].upper() + ext[1:]
                result.append(ext)

            # Add modified sentence if it still has content
            modified_sentence = modified_sentence.strip()
            if modified_sentence:
                # Ensure proper punctuation
                if not any(modified_sentence.endswith(p) for p in ".!?"):
                    modified_sentence += "."
                result.append(modified_sentence)
        else:
            # If no extraction occurred, keep original
            result.append(sentence)

    return result


def normalize_clauses(sentences: List[str]) -> List[str]:
    """Rule 4: Break stacked clauses into sequential sentences for extreme density.

    Splits sentences with multiple nested clauses and excessive punctuation
    into clearer, sequential sentences. Prefers clarity over compactness.
    Never splits decimal numbers, acronyms, or academic intro clauses.

    Args:
        sentences: List of sentence strings

    Returns:
        List of sentences with stacked clauses broken apart
    """
    result = []

    for sentence in sentences:
        word_count = _count_words(sentence)

        # Only process very long sentences (extreme density)
        if word_count < 35:
            result.append(sentence)
            continue

        # Count semicolons as indicators of clause stacking
        semicolon_count = sentence.count(";")

        # If there are semicolons, split at them
        if semicolon_count > 0:
            # Split on semicolons
            segments = re.split(r";\s+", sentence)
            for i, segment in enumerate(segments):
                segment = segment.strip()
                if not segment:
                    continue

                seg_words = _count_words(segment)

                # Only keep segments with sufficient words
                if seg_words >= 5:
                    # Ensure proper punctuation
                    if not any(segment.endswith(p) for p in ".!?"):
                        segment += "."
                    # Normalize as sentence start (capitalize for non-first segments)
                    is_sentence_start = i > 0
                    segment = _normalize_sentence_fragment(segment, is_sentence_start=is_sentence_start)
                    result.append(segment)
                elif result:
                    # Merge short segment with previous (don't capitalize)
                    prev = result[-1].rstrip(".,!?")
                    segment_normalized = _normalize_sentence_fragment(segment, is_sentence_start=False)
                    result[-1] = prev + "; " + segment_normalized

            continue

        # If no semicolons, try splitting at "which" and "that" mid-sentence
        # Only if sentence is very long, and only if not breaking academic intro clauses
        if word_count > 40:
            split_patterns = [r"\b which ", r"\b that "]
            split_occurred = False
            for pattern in split_patterns:
                matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
                if len(matches) > 1:  # Multiple "which"/"that" clauses
                    # Split at second occurrence, but check if it breaks academic clause
                    split_pos = matches[1].start()
                    if _is_academic_intro_clause(sentence, split_pos):
                        continue  # Skip this split
                    
                    first_part = sentence[:split_pos].strip()
                    second_part = sentence[split_pos:].strip()

                    if _count_words(first_part) >= 5 and _count_words(second_part) >= 5:
                        first_part = first_part.rstrip(".,!?") + "."
                        first_part = _normalize_sentence_fragment(first_part, is_sentence_start=True)
                        # Normalize second part as sentence start (capitalize)
                        second_part = second_part.rstrip(".,!?;:")
                        if not second_part.endswith("."):
                            second_part += "."
                        second_part = _normalize_sentence_fragment(second_part, is_sentence_start=True)
                        result.extend([first_part, second_part])
                        split_occurred = True
                        break

            # If splitting occurred, continue to next sentence
            if split_occurred:
                continue

        # If no splitting occurred, keep original
        result.append(sentence)

    return result


def reduce_sentence_count(sentences: List[str], max_avg_length: float = 25.0) -> List[str]:
    """Rule 5: Merge logically adjacent sentences when count is too high.

    Merges sentences to reduce total count, but avoids exceeding normal
    sentence length range. Does not create run-on sentences.

    Args:
        sentences: List of sentence strings
        max_avg_length: Maximum average words per sentence after merging

    Returns:
        List of sentences with some merged
    """
    if len(sentences) <= 4:
        return sentences  # No reduction needed

    result = []
    i = 0

    while i < len(sentences):
        current = sentences[i]
        current_words = _count_words(current)

        # If we're getting close to target count, stop merging
        remaining_needed = max(1, len(sentences) - len(result))
        if len(result) + remaining_needed <= 4:
            # Just append remaining sentences
            result.extend(sentences[i:])
            break

        # Try to merge with next sentence if current is short
        if i < len(sentences) - 1 and current_words < max_avg_length:
            next_sent = sentences[i + 1]
            next_words = _count_words(next_sent)

            combined_words = current_words + next_words

            # Merge if combined length is still reasonable
            if combined_words <= max_avg_length:
                merged = _merge_sentences(current, next_sent)
                result.append(merged)
                i += 2
                continue

        # Keep current sentence without merging
        result.append(current)
        i += 1

    return result


def correct_oversimplification(sentences: List[str]) -> List[str]:
    """Rule 6: Lightly merge adjacent sentences when text is too simple.

    Adds minimal connective phrasing ("which", "that", "and") to create
    slightly more complex sentence structures. Does NOT add new ideas.

    Args:
        sentences: List of sentence strings

    Returns:
        List of sentences with some lightly merged
    """
    if len(sentences) < 2:
        return sentences

    result = []
    i = 0

    while i < len(sentences):
        current = sentences[i]
        current_words = _count_words(current)

        # Only merge if current sentence is short (indicating oversimplification)
        if current_words < 10 and i < len(sentences) - 1:
            next_sent = sentences[i + 1]
            next_words = _count_words(next_sent)

            # Light merge: use "which" or "that" if next sentence could be relative clause
            # Use "and" for coordination
            # Check if next sentence starts with lowercase (might be a continuation)
            if next_sent and next_sent[0].islower():
                # Try "which" connector
                merged = _merge_sentences(current, next_sent, connector=", which")
                result.append(merged)
                i += 2
                continue
            elif current_words + next_words <= 20:  # Keep merged sentences moderate length
                # Use "and" for coordination
                merged = _merge_sentences(current, next_sent, connector=", and")
                result.append(merged)
                i += 2
                continue

        # Keep current sentence without merging
        result.append(current)
        i += 1

    return result


def simplify_complex_sentences(
    sentences: List[str], 
    max_length: int = 20,
    sentence_length_status: str = "normal"
) -> List[str]:
    """Rule 7: Simplify complex sentences when text is too hard to read.

    Handles "too_hard" Flesch scores by:
    1. Extracting parenthetical clauses to reduce complexity
    2. Splitting sentences longer than max_length at natural break points
       ONLY when sentence_length_status is "too_dense" or "extreme_density"

    This rule respects the sentence_length_status gate to prevent splitting
    when sentence length is normal or too_choppy, protecting academic prose.

    Args:
        sentences: List of sentence strings
        max_length: Target maximum words per sentence for readability
        sentence_length_status: Diagnosis status for sentence length.
                                Only splits if "too_dense" or "extreme_density"

    Returns:
        List of sentences with complex ones simplified
    """
    if not sentences:
        return sentences

    # Step 1: Extract parenthetical clauses (reduces complexity)
    # This is safe even when not splitting
    sentences = simplify_parentheses(sentences)

    # Step 2: Only split sentences if sentence_length_status allows it
    # Hard gate: Never split when status is "normal" or "too_choppy"
    if sentence_length_status not in ["too_dense", "extreme_density"]:
        return sentences

    # Step 3: Split long sentences more aggressively (max_length instead of 30)
    result = []
    for sentence in sentences:
        word_count = _count_words(sentence)

        if word_count <= max_length:
            result.append(sentence)
            continue

        # Find potential split points (with academic prose protection)
        split_points = _find_split_points(sentence, protect_academic=True)

        if not split_points:
            # No good split points found, keep as-is
            result.append(sentence)
            continue

        # Try to split at points that create balanced sentences
        current_start = 0
        splits_made = False

        for split_pos in split_points:
            segment = sentence[current_start:split_pos].strip()
            seg_words = _count_words(segment)

            # Only split if segment meets minimum word count
            if seg_words >= 5:
                segment = segment.rstrip(".,!?;:")
                if not segment.endswith("."):
                    segment += "."
                # Normalize as sentence start (capitalize)
                segment = _normalize_sentence_fragment(segment, is_sentence_start=True)
                result.append(segment)
                current_start = split_pos
                splits_made = True

        # Add remaining portion
        if current_start < len(sentence):
            remainder = sentence[current_start:].strip()
            if remainder:
                remainder = remainder.rstrip(".,!?;:")
                if not remainder.endswith("."):
                    remainder += "."
                if _count_words(remainder) >= 5:
                    # Normalize as sentence start (capitalize)
                    remainder = _normalize_sentence_fragment(remainder, is_sentence_start=True)
                    result.append(remainder)
                elif splits_made and result:
                    # Merge short remainder with previous (don't capitalize)
                    remainder_normalized = _normalize_sentence_fragment(
                        remainder.rstrip(".,!?;:"), 
                        is_sentence_start=False
                    )
                    result[-1] = result[-1].rstrip(".,!?;:") + ", " + remainder_normalized
                else:
                    # Normalize as sentence start
                    remainder = _normalize_sentence_fragment(remainder, is_sentence_start=True)
                    result.append(remainder)

        # If no splits were made, keep original
        if not splits_made:
            result.append(sentence)

    return result


class RewriteEngine:
    """Coordinator for applying rewrite rules based on text diagnosis.

    The engine analyzes text, diagnoses issues, and applies appropriate
    rewrite rules in a documented order. Returns text unchanged if all
    diagnoses are "normal".
    """

    def __init__(self, text: str):
        """Initialize rewrite engine with text to process.

        Args:
            text: Raw text string to analyze and potentially rewrite
        """
        self.text = text
        self.analyzer = TextAnalyzer(text)
        self.diagnosis = TextDiagnosis(self.analyzer)

    def rewrite(self) -> str:
        """Apply rewrite rules based on diagnosis and return rewritten text.

        Rules are applied in the following order:
        1. Sentence structure rules (merge/split) for length issues
        2. Parentheses and clause normalization for density issues
        3. Sentence count reduction
        4. Over-simplification correction

        Hard early-exit gate: If ALL diagnosis statuses are "normal",
        returns original text unchanged before ANY rewrite rule is applied.
        This protects well-formed academic prose from unnecessary modifications.

        Returns:
            Rewritten text string, or original text if no changes needed
        """
        # Get diagnosis
        diag = self.diagnosis.diagnose()

        # Hard early-exit gate: If ALL diagnosis statuses are "normal",
        # return original text unchanged before ANY rewrite rule is applied.
        # This is critical for protecting well-formed academic prose that may
        # have slightly unusual metrics but is still grammatically correct.
        if (
            diag.flesch_status == "normal"
            and diag.sentence_length_status == "normal"
            and diag.sentence_count_status == "normal"
        ):
            return self.text

        # Start with current sentences
        sentences = self.analyzer.sentences.copy()

        # Apply rules in documented order based on diagnosis

        # Rule 1: Merge short sentences (too_choppy)
        if diag.sentence_length_status == "too_choppy":
            sentences = merge_short_sentences(sentences)

        # Rule 2: Split long sentences (too_dense or extreme_density)
        # Sentence splitting is ONLY allowed when sentence_length_status
        # is "too_dense" or "extreme_density". This prevents splitting
        # of normal or too_choppy text, protecting academic prose.
        if diag.sentence_length_status in ["too_dense", "extreme_density"]:
            sentences = split_long_sentences(sentences)

        # Rule 3: Parentheses simplification (too_dense or extreme_density)
        if diag.sentence_length_status in ["too_dense", "extreme_density"]:
            sentences = simplify_parentheses(sentences)

        # Rule 4: Clause normalization (extreme_density only)
        if diag.sentence_length_status == "extreme_density":
            sentences = normalize_clauses(sentences)

        # Rule 5: Sentence count reduction (too_many_sentences)
        if diag.sentence_count_status == "too_many_sentences":
            sentences = reduce_sentence_count(sentences)

        # Rule 6: Over-simplification correction (too_simple)
        if diag.flesch_status == "too_simple":
            sentences = correct_oversimplification(sentences)

        # Rule 7: Simplify complex sentences (too_hard)
        # Only splits if sentence_length_status is "too_dense" or "extreme_density"
        # This protects academic prose from unnecessary splitting when sentence
        # length is already normal, even if Flesch score indicates difficulty.
        if diag.flesch_status == "too_hard":
            sentences = simplify_complex_sentences(
                sentences, 
                sentence_length_status=diag.sentence_length_status
            )

        # Reassemble text from sentences
        # Join with spaces, ensure single space between sentences
        rewritten = " ".join(sentences)

        # Normalize whitespace (multiple spaces to single space)
        rewritten = re.sub(r"\s+", " ", rewritten)

        # Ensure space after sentence-ending punctuation
        rewritten = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", rewritten)

        # Post-processing safety checks:
        # 1. Ensure all sentences start with capital letters
        # Split on sentence boundaries and capitalize each sentence start
        sentence_boundary_pattern = r"([.!?])\s+([a-z])"
        def capitalize_sentence_start(match):
            return match.group(1) + " " + match.group(2).upper()
        rewritten = re.sub(sentence_boundary_pattern, capitalize_sentence_start, rewritten)
        
        # 2. Capitalize first letter of entire text if it's lowercase
        if rewritten and rewritten[0].islower():
            rewritten = rewritten[0].upper() + rewritten[1:]
        
        # 3. Remove duplicate punctuation (but preserve intentional multiple punctuation)
        rewritten = re.sub(r"([.!?])\1+", r"\1", rewritten)
        
        # 4. Ensure no missing spaces after periods
        rewritten = re.sub(r"\.([A-Za-z])", r". \1", rewritten)

        return rewritten.strip()
