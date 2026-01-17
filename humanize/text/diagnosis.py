"""Text diagnosis module for determining if text falls outside human writing ranges."""

from dataclasses import dataclass
from typing import Dict

from humanize.text.analyzer import TextAnalyzer


@dataclass
class DiagnosisResult:
    """Structured diagnosis result for text analysis."""

    flesch_status: str
    sentence_length_status: str
    sentence_count_status: str

    def to_dict(self) -> Dict[str, str]:
        """Convert diagnosis result to dictionary."""
        return {
            "flesch_status": self.flesch_status,
            "sentence_length_status": self.sentence_length_status,
            "sentence_count_status": self.sentence_count_status,
        }


class TextDiagnosis:
    """Determines if text falls outside human writing ranges based on precomputed metrics.

    Uses a TextAnalyzer instance to access text metrics and evaluates them against
    empirically derived human writing ranges. Does not modify text or perform rewriting.
    """

    def __init__(self, analyzer: TextAnalyzer):
        """Initialize with a TextAnalyzer instance.

        Args:
            analyzer: TextAnalyzer instance providing text metrics
        """
        self.analyzer = analyzer

    def diagnose(self) -> DiagnosisResult:
        """Generate diagnosis result for the analyzed text.

        Returns:
            DiagnosisResult containing status for each evaluated metric
        """
        flesch_status = self._diagnose_flesch()
        sentence_length_status = self._diagnose_sentence_length()
        sentence_count_status = self._diagnose_sentence_count()

        return DiagnosisResult(
            flesch_status=flesch_status,
            sentence_length_status=sentence_length_status,
            sentence_count_status=sentence_count_status,
        )

    def _diagnose_flesch(self) -> str:
        """Diagnose Flesch Reading Ease score.

        Returns:
            Status: "too_hard", "normal", or "too_simple"
        """
        flesch_score = self.analyzer.flesch_score

        if flesch_score is None:
            return "normal"  # Default when score cannot be calculated

        if flesch_score < 40:
            return "too_hard"
        elif flesch_score <= 90:
            return "normal"
        else:
            return "too_simple"

    def _diagnose_sentence_length(self) -> str:
        """Diagnose average sentence length.

        Returns:
            Status: "too_choppy", "normal", "too_dense", or "extreme_density"
        """
        if self.analyzer.sentence_count == 0:
            return "normal"  # Default when no sentences

        avg_sentence_length = self.analyzer.word_count / self.analyzer.sentence_count

        if avg_sentence_length < 5:
            return "too_choppy"
        elif avg_sentence_length <= 25:
            return "normal"
        elif avg_sentence_length > 40:
            return "extreme_density"
        else:
            return "too_dense"

    def _diagnose_sentence_count(self) -> str:
        """Diagnose sentence count.

        Returns:
            Status: "normal" or "too_many_sentences"
        """
        if self.analyzer.sentence_count <= 4:
            return "normal"
        else:
            return "too_many_sentences"

    def summary(self) -> str:
        """Generate human-readable summary of diagnosis.

        Returns:
            Multi-line string describing the diagnosis results
        """
        result = self.diagnose()

        lines = ["Text Diagnosis Summary", "=" * 50]
        lines.append(f"Flesch Reading Ease: {result.flesch_status}")
        lines.append(f"Average Sentence Length: {result.sentence_length_status}")
        lines.append(f"Sentence Count: {result.sentence_count_status}")

        return "\n".join(lines)
