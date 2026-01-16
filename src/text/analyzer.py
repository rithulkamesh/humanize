import re
from typing import List, Optional


class TextAnalyzer:
    """Analyzes text and provides readability metrics using Flesch Reading Ease."""

    def __init__(self, text: str):
        """Initialize with text to analyze.

        Args:
            text: The text string to analyze
        """
        self.text = text
        self._sentences: Optional[List[str]] = None
        self._word_count: Optional[int] = None
        self._syllable_count: Optional[int] = None
        self._flesch_score: Optional[float] = None

    def _split_sentences(self) -> List[str]:
        """Split text into sentences using regex pattern.

        Returns:
            List of sentence strings, with empty/trivial sentences filtered out
        """
        # Normalize whitespace and replace newlines with spaces
        normalized = re.sub(r"\s+", " ", self.text)
        normalized = normalized.replace("\n", " ")

        # Split on sentence boundaries using lookbehind for .!?
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, normalized)

        # Strip whitespace and filter out empty/trivial sentences
        result = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        return result

    @property
    def sentences(self) -> List[str]:
        """List of sentences extracted from the text."""
        if self._sentences is None:
            self._sentences = self._split_sentences()
        return self._sentences

    @property
    def sentence_count(self) -> int:
        """Number of sentences in the text."""
        return len(self.sentences)

    def _split_words(self) -> List[str]:
        """Split text into words, handling punctuation and numbers.

        Returns:
            List of word strings with punctuation stripped
        """
        # Split on whitespace
        words = re.split(r"\s+", self.text)

        result = []
        for word in words:
            # Strip leading/trailing punctuation
            cleaned = word.strip(".,!?;:()[]{}'\"-—_")
            # Include if not empty (numbers count as words)
            if cleaned:
                result.append(cleaned)

        return result

    def _count_words(self) -> int:
        """Count total words in the text."""
        return len(self._split_words())

    @property
    def word_count(self) -> int:
        """Total number of words in the text."""
        if self._word_count is None:
            self._word_count = self._count_words()
        return self._word_count

    def _count_syllables_in_word(self, word: str) -> int:
        """Count syllables in a single word using heuristic rules.

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
        # Pattern matches consecutive vowels as a single group
        vowel_groups = re.findall(r"[aeiouy]+", cleaned)
        syllable_count = len(vowel_groups)

        # Subtract one syllable if word ends with silent 'e'
        if cleaned.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        # Minimum 1 syllable per word
        return max(1, syllable_count)

    def _count_total_syllables(self) -> int:
        """Count total syllables across all words in the text."""
        words = self._split_words()
        return sum(self._count_syllables_in_word(word) for word in words)

    @property
    def syllable_count(self) -> int:
        """Total number of syllables in the text."""
        if self._syllable_count is None:
            self._syllable_count = self._count_total_syllables()
        return self._syllable_count

    def _calculate_flesch_score(self) -> Optional[float]:
        """Calculate Flesch Reading Ease score.

        Formula: 206.835 - (1.015 * ASL) - (84.6 * ASW)
        Where ASL = average sentence length (words/sentences)
        and ASW = average syllables per word (syllables/words)

        Returns:
            Score between 0-100, or None if calculation not possible
        """
        if self.sentence_count == 0 or self.word_count == 0:
            return None

        # Calculate average sentence length
        asl = self.word_count / self.sentence_count

        # Calculate average syllables per word
        asw = self.syllable_count / self.word_count

        # Flesch formula
        score = 206.835 - (1.015 * asl) - (84.6 * asw)

        # Clamp between 0 and 100
        return max(0.0, min(100.0, score))

    @property
    def flesch_score(self) -> Optional[float]:
        """Flesch Reading Ease score (0-100, higher is easier to read).

        Returns None if text has no sentences or words.
        """
        if self._flesch_score is None:
            self._flesch_score = self._calculate_flesch_score()
        return self._flesch_score
