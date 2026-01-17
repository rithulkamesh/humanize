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


def _merge_sentences(s1: str, s2: str, connector: str = ", and") -> str:
    """Merge two sentences with a connector.

    Args:
        s1: First sentence
        s2: Second sentence
        connector: Connector to use between sentences

    Returns:
        Merged sentence
    """
    s1_clean = s1.rstrip(".,!?;:")
    s2_clean = s2.lstrip(".,!?;:")
    return f"{s1_clean}{connector} {s2_clean}"


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


def _find_split_points(sentence: str) -> List[int]:
    """Find potential split points in a sentence (commas, conjunctions).

    Args:
        sentence: The sentence to analyze

    Returns:
        List of character positions where sentence could be split
    """
    points = []
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
                points.append(match.end())

    # Sort and deduplicate
    points = sorted(set(points))
    return points


def split_long_sentences(sentences: List[str], min_words: int = 5) -> List[str]:
    """Rule 2: Split long sentences at commas and coordinating conjunctions.

    Splits sentences that are too long at natural break points (commas,
    conjunctions like "and", "but", "which", "that"), ensuring resulting
    sentences have at least min_words words.

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

        # Find potential split points
        split_points = _find_split_points(sentence)

        if not split_points:
            # No good split points found, keep as-is
            result.append(sentence)
            continue

        # Try to split at points that create balanced sentences
        current_start = 0
        for split_pos in split_points:
            # Extract segment before split point
            segment = sentence[current_start:split_pos].strip()
            seg_words = _count_words(segment)

            # Only split if segment meets minimum word count
            if seg_words >= min_words:
                # Remove trailing punctuation from segment, add period
                segment = segment.rstrip(".,!?;:")
                if not segment.endswith("."):
                    segment += "."
                result.append(segment)
                current_start = split_pos

        # Add remaining portion
        if current_start < len(sentence):
            remainder = sentence[current_start:].strip()
            if remainder:
                remainder = remainder.rstrip(".,!?;:")
                if not remainder.endswith("."):
                    remainder += "."
                # Only add if it meets minimum
                if _count_words(remainder) >= min_words:
                    result.append(remainder)
                else:
                    # Merge with previous sentence if too short
                    if result:
                        result[-1] = result[-1].rstrip(".") + ", " + remainder.rstrip(".")
                    else:
                        result.append(remainder)

        # If no splits were made, keep original
        if current_start == 0:
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

        # If no extraction occurred, keep original
        if not extracted:
            result.append(sentence)

    return result


def normalize_clauses(sentences: List[str]) -> List[str]:
    """Rule 4: Break stacked clauses into sequential sentences for extreme density.

    Splits sentences with multiple nested clauses and excessive punctuation
    into clearer, sequential sentences. Prefers clarity over compactness.

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
                    # Capitalize if it's not the first segment
                    if i > 0 and segment and segment[0].islower():
                        segment = segment[0].upper() + segment[1:]
                    result.append(segment)
                elif result:
                    # Merge short segment with previous
                    prev = result[-1].rstrip(".,!?")
                    result[-1] = prev + "; " + segment

            continue

        # If no semicolons, try splitting at "which" and "that" mid-sentence
        # Only if sentence is very long
        if word_count > 40:
            split_patterns = [r"\b which ", r"\b that "]
            split_occurred = False
            for pattern in split_patterns:
                matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
                if len(matches) > 1:  # Multiple "which"/"that" clauses
                    # Split at second occurrence
                    split_pos = matches[1].start()
                    first_part = sentence[:split_pos].strip()
                    second_part = sentence[split_pos:].strip()

                    if _count_words(first_part) >= 5 and _count_words(second_part) >= 5:
                        first_part = first_part.rstrip(".,!?") + "."
                        # Capitalize second part if needed
                        if second_part and second_part[0].islower():
                            second_part = second_part[0].upper() + second_part[1:]
                        if not second_part.endswith("."):
                            second_part += "."
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

        Returns:
            Rewritten text string, or original text if no changes needed
        """
        # Get diagnosis
        diag = self.diagnosis.diagnose()

        # If everything is normal, return unchanged
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

        # Reassemble text from sentences
        # Join with spaces, ensure single space between sentences
        rewritten = " ".join(sentences)

        # Normalize whitespace (multiple spaces to single space)
        rewritten = re.sub(r"\s+", " ", rewritten)

        # Ensure space after sentence-ending punctuation
        rewritten = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", rewritten)

        return rewritten.strip()
