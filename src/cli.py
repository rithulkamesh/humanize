from argparse import ArgumentParser
from flesch import calculate_flesh_score
import sys


def main():
    parser = ArgumentParser(
        prog="humanize",
        description="A rule-based system for humanizing text using open datasets and deterministic transformations.",
        epilog="For more information, visit: https://github.com/rithulkamesh/humanize",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # humanize command
    humanize_parser = subparsers.add_parser(
        "humanize",
        help="Humanize text by applying rule-based transformations",
    )
    humanize_parser.add_argument(
        "text",
        nargs="?",
        help="Text to humanize (reads from stdin if not provided)",
    )
    humanize_parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to input text file",
    )
    humanize_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (writes to stdout if not provided)",
    )
    humanize_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable optional local LLM refinement",
    )

    # version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "humanize":
        handle_humanize(args)
    elif args.command == "info":
        handle_info()
    elif args.command == "version":
        handle_version()


def handle_info():
    """Display system information."""
    info_text = """
Humanize - Rule-Based Text Humanization

Features:
  • Analyzes structural features like sentence length and rhythm
  • Compares against open reference examples of human writing
  • Applies deterministic, rule-based transformations
  • Preserves original meaning while improving naturalness
  • Fully offline and transparent

What it does NOT do:
  • Does not generate text from scratch
  • Does not require API keys or paid services
  • Does not train models on user input
  • Does not require internet connection

Optional Features:
  • Local language model support (disabled by default)
  • Deterministic transformation pipeline
  • Open datasets for reference data

License: GNU General Public License v3 (GPLv3)
"""
    print(info_text)


def handle_version():
    """Display version information."""
    print("humanize version 0.1.0")


def handle_humanize(args):
    """Process text humanization request."""
    # Get input text
    if args.file:
        try:
            with open(args.file, "r") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        print("Error: No input provided. Use -f/--file or pass text as argument.", file=sys.stderr)
        sys.exit(1)

    # TODO: Implement humanization logic

    out_data = {"text": text, "flesch_score": calculate_flesh_score(text)}

    print(out_data)
    sys.exit(1)


if __name__ == "__main__":
    main()
