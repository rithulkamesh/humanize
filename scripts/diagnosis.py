import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import humanize.text.diagnosis as diagnosis
import humanize.text.analyzer as analyzer


def main():
    text = "Yes. You’re understanding this exactly right. At this point, stop adding code."
    ana = analyzer.TextAnalyzer(text)
    dia = diagnosis.TextDiagnosis(ana)
    print(dia.summary())


if __name__ == "__main__":
    main()
