"""Deterministic rule-based lexical simplification for vocabulary complexity reduction.

This module implements LexicalSimplifier, which reduces vocabulary complexity
using word frequency analysis from the writing_events dataset. Simplification
is conservative, opt-in, and strictly data-derived from dataset statistics.

The simplifier never modifies technical terms, acronyms, or domain-specific
nouns, and includes hard safety guards to protect academic text.
"""

from humanize.data.dataset import HumanizeDataset
from humanize.text.analyzer import TextAnalyzer

import re
from typing import Dict, List, Optional, Tuple

import polars as pl


def count_syllables_in_word(word: str) -> int:
    """Count syllables in a single word using TextAnalyzer's heuristic rules.

    Args:
        word: The word to count syllables for

    Returns:
        Number of syllables (minimum 1)
    """
    # Lowercase and remove non-alphabetic characters
    cleaned = re.sub(r"[^a-zA-Z]", "", word.lower())

    if not cleaned:
        return 1

    # Count vowel groups (a, e, i, o, u, y)
    vowel_groups = re.findall(r"[aeiouy]+", cleaned)
    syllable_count = len(vowel_groups)

    # Subtract one syllable if word ends with silent 'e'
    if cleaned.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    # Minimum 1 syllable per word
    return max(1, syllable_count)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using TextAnalyzer's logic.

    Args:
        text: The text to split

    Returns:
        List of sentence strings
    """
    # Normalize whitespace and replace newlines with spaces
    normalized = re.sub(r"\s+", " ", text)
    normalized = normalized.replace("\n", " ")

    # Split on sentence boundaries using lookbehind for .!?
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, normalized)

    # Strip whitespace and filter out empty/trivial sentences
    result = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
    return result


def split_words(text: str) -> List[str]:
    """Split text into words, handling punctuation.

    Args:
        text: The text to tokenize

    Returns:
        List of word strings with punctuation stripped
    """
    # Split on whitespace
    words = re.split(r"\s+", text)

    result = []
    for word in words:
        # Strip leading/trailing punctuation
        cleaned = word.strip(".,!?;:()[]{}'\"-—_")
        # Include if not empty
        if cleaned:
            result.append(cleaned)

    return result


def is_acronym(word: str) -> bool:
    """Check if a word is an acronym (ALL CAPS, 2+ characters).

    Args:
        word: The word to check

    Returns:
        True if the word is an acronym, False otherwise
    """
    return bool(re.match(r"^[A-Z]{2,}$", word))


def is_number(word: str) -> bool:
    """Check if a word consists entirely of digits.

    Args:
        word: The word to check

    Returns:
        True if the word is a number, False otherwise
    """
    return bool(re.match(r"^\d+$", word))


def find_parentheses_ranges(text: str) -> List[Tuple[int, int]]:
    """Find all ranges of text inside parentheses.

    Args:
        text: The text to search

    Returns:
        List of (start, end) tuples for text inside parentheses
    """
    ranges = []
    stack = []

    for i, char in enumerate(text):
        if char == "(":
            stack.append(i)
        elif char == ")" and stack:
            start = stack.pop()
            ranges.append((start, i + 1))

    return ranges


def is_inside_parentheses(pos: int, ranges: List[Tuple[int, int]]) -> bool:
    """Check if a position is inside any parentheses range.

    Args:
        pos: Character position to check
        ranges: List of (start, end) tuples for parentheses ranges

    Returns:
        True if position is inside parentheses, False otherwise
    """
    for start, end in ranges:
        if start <= pos < end:
            return True
    return False


def would_create_awkward_grammar(
    original: str, replacement: str, sentence: str, word_start: int
) -> bool:
    """Check if replacing a word would create awkward grammar.

    Args:
        original: The original word
        replacement: The proposed replacement
        sentence: The sentence containing the word
        word_start: Character position where word starts

    Returns:
        True if replacement would create awkward grammar, False otherwise
    """
    # Check for repeated words (e.g., "the the")
    after_start = word_start + len(original)
    after = sentence[after_start:].lstrip()
    before = sentence[:word_start].rstrip()

    if before:
        last_word_before = before.split()[-1].lower()
        if after:
            first_word_after = after.split()[0].lower() if after.split() else ""
            if last_word_before == replacement.lower() or replacement.lower() == first_word_after:
                return True

    return False


def is_likely_proper_noun(word: str, sentence: str, word_start: int) -> bool:
    """Heuristic check if a word is likely a proper noun.

    Checks if word is capitalized mid-sentence (not at sentence start),
    which suggests it's a proper noun.

    Args:
        word: The word to check
        sentence: The sentence containing the word
        word_start: Character position where word starts in sentence

    Returns:
        True if word is likely a proper noun, False otherwise
    """
    # Check if word starts with uppercase
    if not word[0].isupper():
        return False

    # Check if word is at sentence start (before this position, only whitespace/punctuation)
    before_text = sentence[:word_start].strip()
    # If nothing before (or only punctuation), it's sentence start
    if not before_text or before_text.endswith((".", "!", "?", ":", ";")):
        return False

    # Capitalized mid-sentence is likely a proper noun
    return True


class LexicalSimplifier:
    """Lexical simplifier that reduces vocabulary complexity using word frequency analysis.

    The simplifier uses strictly data-derived replacements from the word frequency dataset
    to identify and replace rare, complex words with more common alternatives. All
    replacement decisions are computed dynamically from dataset statistics.

    Safety constraints:
    - Never replaces technical terms, acronyms, or domain-specific nouns
    - Maximum 10 replacements per paragraph (configurable)
    - Never replaces words inside parentheses
    - Never replaces proper nouns
    - Only replaces one word per sentence by default
    - Never replaces words above 60th percentile frequency (common words)
    """

    def __init__(
        self,
        dataset: HumanizeDataset,
        percentile_threshold: float = 0.6,
        max_replacements_per_paragraph: int = 10,
        max_replacements_per_sentence: int = 1,
    ):
        """Initialize lexical simplifier.

        Args:
            dataset: HumanizeDataset containing all required runtime data
            percentile_threshold: Words below this percentile are considered rare (default: 0.6 = bottom 60%)
            max_replacements_per_paragraph: Maximum replacements per paragraph (default: 10)
            max_replacements_per_sentence: Maximum replacements per sentence (default: 1)
        """
        self.percentile_threshold = percentile_threshold
        self.max_replacements_per_paragraph = max_replacements_per_paragraph
        self.max_replacements_per_sentence = max_replacements_per_sentence

        # Extract lexical_stats from dataset (guaranteed non-None)
        self.frequency_table: pl.DataFrame = dataset.lexical_stats

        # Build lookup dictionaries
        self.frequency_dict: Dict[str, int] = {}
        self.syllable_dict: Dict[str, int] = {}

        for row in self.frequency_table.iter_rows(named=True):
            word = row["word"]
            freq = row["frequency"]
            syllables = row["syllable_count"]

            self.frequency_dict[word] = freq
            self.syllable_dict[word] = syllables

        # Compute percentile threshold
        if len(self.frequency_table) > 0:
            # Get frequencies sorted ascending
            frequencies = sorted(self.frequency_dict.values())
            threshold_index = int(len(frequencies) * self.percentile_threshold)
            self.percentile_frequency_threshold = (
                frequencies[threshold_index]
                if threshold_index < len(frequencies)
                else frequencies[-1]
            )

            # Mark words below threshold as rare (candidates for replacement)
            # Words above threshold are common and should not be replaced
            self.rare_words = {
                word
                for word, freq in self.frequency_dict.items()
                if freq <= self.percentile_frequency_threshold
            }
        else:
            # Empty dataset - set defaults
            self.percentile_frequency_threshold = 0
            self.rare_words = set()

    def _estimate_pos(self, word: str, sentence: str, word_start: int) -> Optional[str]:
        """Estimate part of speech for a word using simple heuristics.

        This is a conservative heuristic-based POS detector that protects nouns
        by default. Nouns are protected because they often contain technical terms
        and domain concepts that should not be replaced in academic text.

        Args:
            word: The word to analyze
            sentence: The sentence containing the word
            word_start: Character position where word starts in sentence

        Returns:
            Estimated POS: "noun", "verb", "adjective", "adverb", or None
        """
        word_lower = word.lower()

        # Check for verb endings
        if word_lower.endswith(("ing", "ed", "ize", "ise")):
            # Check if it's actually an adjective (e.g., "interesting", "complicated")
            if word_lower.endswith("ing") and len(word_lower) > 5:
                # Could be adjective (e.g., "interesting") or verb (e.g., "working")
                # Be conservative: if it ends in -ing and is long, might be adjective
                # For now, treat as verb/adjective (not noun)
                return "verb" if not word_lower.endswith(("ring", "ling")) else "adjective"
            return "verb"

        # Check for adjective endings
        if word_lower.endswith(("ive", "ous", "ful", "less", "able", "ible", "ic", "al")):
            return "adjective"

        # Check for adverb endings
        if word_lower.endswith("ly"):
            return "adverb"

        # Check for past participle (often used as adjective)
        if word_lower.endswith(("ed", "en")) and len(word_lower) > 4:
            return "verb"

        # Default: treat as noun (most common, and we want to protect nouns)
        # Nouns are protected because they often contain technical terms and domain
        # concepts that should not be replaced in academic and technical prose
        return "noun"

    def _is_candidate_for_simplification(
        self, word: str, syllable_count: int, word_lower: str, sentence: str, word_start: int
    ) -> bool:
        """Check if a word is a candidate for simplification.

        A word is a candidate if:
        - syllable_count >= 4 AND word is rare (frequency below 60th percentile)
        - NOT an acronym
        - NOT a number
        - NOT a noun (nouns are protected to preserve technical terms and domain concepts)
        - Only verbs and adjectives are candidates

        Args:
            word: The original word (preserving case)
            syllable_count: Number of syllables in the word
            word_lower: Lowercase version of the word
            sentence: The sentence containing the word (for POS estimation)
            word_start: Character position where word starts in sentence

        Returns:
            True if word is a candidate for simplification, False otherwise
        """
        # Filter out acronyms
        if is_acronym(word):
            return False

        # Filter out numbers
        if is_number(word):
            return False

        # Filter out hyphenated multi-token words (e.g., "high-frequency", "built-in")
        # These should be preserved as they are technical terms or compound concepts
        if "-" in word and len(word.split("-")) > 1:
            return False

        # Filter out words adjacent to numbers (e.g., "version 2.0", "3D model")
        # Check context around word for adjacent numbers
        before_context = sentence[:word_start].strip()
        after_context = sentence[word_start + len(word) :].strip()
        if re.search(r"\d+", before_context[-10:]) or re.search(r"^\d+", after_context[:10]):
            return False

        # POS filtering: only allow verbs and adjectives for replacement
        # Nouns are protected because they often contain technical terms and domain
        # concepts that should not be replaced in academic and technical prose
        pos = self._estimate_pos(word, sentence, word_start)
        if pos == "noun":
            return False  # Never replace nouns (preserve technical terms)

        # Only allow verbs and adjectives for replacement
        if pos not in ("verb", "adjective"):
            return False

        # Must have at least 4 syllables
        if syllable_count < 4:
            return False

        # Check if word is in dataset
        if word_lower not in self.frequency_dict:
            # Word not in dataset - consider it rare and eligible for replacement
            # (We can't find a replacement for it, but this allows other processing)
            return True

        # Check if rare (frequency below 60th percentile)
        # Words above 60th percentile are common and should not be replaced
        if word_lower not in self.rare_words:
            return False  # Word is too common

        return True

    def _find_replacement(
        self, word_lower: str, syllable_count: int, pos: str, sentence: str, word_start: int
    ) -> Optional[str]:
        """Find a replacement for a word by searching the frequency dataset.

        Searches for candidates matching all safety criteria:
        - Same POS (verb/adjective)
        - Fewer syllables than original
        - Similar length (within ±2 characters)
        - Higher frequency than original
        - In dataset vocabulary
        - Above 60th percentile frequency (common enough to be safe)

        Selection is deterministic: candidates are sorted by frequency (descending),
        then by syllable count (ascending), then by length (ascending), then alphabetically.

        Args:
            word_lower: Lowercase word to find replacement for
            syllable_count: Syllable count of the original word
            pos: Part of speech of the original word
            sentence: The sentence containing the word (for POS estimation of candidates)
            word_start: Character position where word starts in sentence

        Returns:
            Replacement word (lowercase) if found and safe, None otherwise
        """
        if word_lower not in self.frequency_dict:
            return None

        original_freq = self.frequency_dict[word_lower]
        original_length = len(word_lower)

        # Find all candidate replacements from the dataset
        candidates = []
        for candidate_word, candidate_freq in self.frequency_dict.items():
            # Skip the original word
            if candidate_word == word_lower:
                continue

            # Candidate must be in dataset
            if candidate_word not in self.syllable_dict:
                continue

            # Candidate must have higher frequency (more common)
            if candidate_freq <= original_freq:
                continue

            # Candidate must be above 60th percentile (common enough)
            if candidate_freq <= self.percentile_frequency_threshold:
                continue

            # Candidate must have fewer syllables
            candidate_syllables = self.syllable_dict[candidate_word]
            if candidate_syllables >= syllable_count:
                continue

            # Candidate must have similar length (within ±2 characters)
            if abs(len(candidate_word) - original_length) > 2:
                continue

            # Candidate must have same POS (heuristic check)
            # Estimate POS for candidate using a simple sentence context
            candidate_pos = self._estimate_pos(candidate_word, sentence, word_start)
            if candidate_pos != pos:
                continue

            candidates.append((candidate_word, candidate_freq, candidate_syllables))

        if not candidates:
            return None

        # Sort deterministically:
        # 1. Frequency (descending) - prefer more common words
        # 2. Syllable count (ascending) - prefer simpler words
        # 3. Length (ascending) - prefer shorter words
        # 4. Alphabetical - for tie-breaking
        candidates.sort(key=lambda x: (-x[1], x[2], len(x[0]), x[0]))

        # Return the best candidate
        return candidates[0][0]

    def _simplify_sentence(self, sentence: str) -> str:
        """Simplify a single sentence with strict limits and safety checks.

        Conservative simplification with maximum 1 replacement per sentence.
        Skips already-readable sentences (Flesch >= 50) to preserve academic tone.

        Args:
            sentence: The sentence to simplify

        Returns:
            Simplified sentence
        """
        words = split_words(sentence)
        if not words:
            return sentence

        # Flesch score check: skip already-readable sentences
        # Sentences with Flesch >= 50 are already readable and don't need
        # lexical simplification, preserving academic tone and precision
        analyzer = TextAnalyzer(sentence)
        flesch_score = analyzer.flesch_score

        if flesch_score is not None and flesch_score >= 50:
            return sentence  # Already readable, skip simplification

        result = sentence
        sentence_replacements = 0  # Track replacements per sentence (max 1)

        # Find parentheses ranges within this sentence
        sentence_parentheses = find_parentheses_ranges(result)

        # Collect all replacement candidates (word, replacement pairs)
        replacements_to_apply = []

        for i, word in enumerate(words):
            # Check sentence replacement limit (max 1 per sentence)
            if sentence_replacements >= self.max_replacements_per_sentence:
                break  # Stop immediately when limit reached

            word_lower = word.lower()
            syllable_count = count_syllables_in_word(word)

            # Find word position in result
            word_start_in_sentence = result.find(word)
            if word_start_in_sentence < 0:
                continue

            # Skip if not a candidate (includes POS filtering)
            if not self._is_candidate_for_simplification(
                word, syllable_count, word_lower, result, word_start_in_sentence
            ):
                continue

            # Skip if inside parentheses
            inside_parens = False
            for paren_start, paren_end in sentence_parentheses:
                if paren_start <= word_start_in_sentence < paren_end:
                    inside_parens = True
                    break
            if inside_parens:
                continue

            # Skip if likely proper noun
            if is_likely_proper_noun(word, result, word_start_in_sentence):
                continue

            # Estimate POS for replacement search
            pos = self._estimate_pos(word, result, word_start_in_sentence)

            # Try to find replacement
            replacement = self._find_replacement(
                word_lower, syllable_count, pos, result, word_start_in_sentence
            )

            if replacement:
                # Preserve original capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                if word.isupper():
                    replacement = replacement.upper()

                # Check if replacement would create awkward grammar (repeated words)
                if would_create_awkward_grammar(word, replacement, result, word_start_in_sentence):
                    continue  # Skip if would create awkward grammar

                replacements_to_apply.append((word, replacement, word_start_in_sentence))

        # Apply word replacements (in reverse order to preserve positions)
        # But stop at sentence replacement limit
        for word, replacement, word_start_in_sentence in reversed(replacements_to_apply):
            # Check sentence replacement limit before applying
            if sentence_replacements >= self.max_replacements_per_sentence:
                break  # Stop immediately when limit reached

            # Replace word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(word) + r"\b"
            result = re.sub(pattern, replacement, result, count=1)
            sentence_replacements += 1

        return result

    def simplify(self, text: str) -> str:
        """Simplify text by replacing rare, complex words with common alternatives.

        Conservative simplification with strict limits:
        - Maximum 1 replacement per sentence
        - Maximum 10 replacements per paragraph (configurable)
        - Preserves academic tone and technical accuracy
        - All replacements are data-derived from frequency dataset

        Args:
            text: The text to simplify

        Returns:
            Simplified text, or original text if no changes needed
        """

        # Split text into paragraphs (preserve paragraph breaks)
        # Paragraphs are separated by \n\n or \n\s*\n
        paragraphs = re.split(r"\n\s*\n", text)
        simplified_paragraphs = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                simplified_paragraphs.append(paragraph)
                continue

            # Track replacements per paragraph (max 1 per paragraph)
            # This preserves academic tone by limiting changes per paragraph
            paragraph_replacements = 0

            # Split paragraph into sentences
            sentences = split_sentences(paragraph)
            if not sentences:
                simplified_paragraphs.append(paragraph)
                continue

            # Process each sentence in the paragraph
            simplified_sentences = []
            for sentence in sentences:
                # Check paragraph replacement limit (max 10 per paragraph)
                # Stop immediately when limit reached to preserve academic tone
                if paragraph_replacements >= self.max_replacements_per_paragraph:
                    simplified_sentences.append(sentence)
                    continue

                simplified = self._simplify_sentence(sentence)

                # Count replacements made in this sentence
                # Each sentence can have at most 1 replacement, so we count by sentence change
                if simplified != sentence:
                    paragraph_replacements += 1

                simplified_sentences.append(simplified)

            # Rejoin sentences with spaces (preserves punctuation)
            simplified_paragraph = " ".join(simplified_sentences)
            simplified_paragraphs.append(simplified_paragraph)

        # Rejoin paragraphs with double newlines (preserves paragraph structure)
        return "\n\n".join(simplified_paragraphs)
