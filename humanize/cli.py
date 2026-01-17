from argparse import ArgumentParser
import sys
from pathlib import Path

# Handle import for both module and script execution
try:
    from humanize import humanize
    from humanize.data.dataset import HumanizeDataset
    from humanize.text.simplify import LexicalSimplifier
except ImportError:
    # When running as a script, add parent directory to path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from humanize import humanize
    from humanize.data.dataset import HumanizeDataset
    from humanize.text.simplify import LexicalSimplifier


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
        "--lexical",
        action="store_true",
        help="Enable lexical simplification (reduces vocabulary complexity)",
    )
    humanize_parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/humanize.parquet",
        help="Path to HumanizeDataset bundle file (default: datasets/humanize.parquet)",
    )

    # version command
    subparsers.add_parser(
        "version",
        help="Show version information",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "humanize":
        handle_humanize(args)
    elif args.command == "version":
        handle_version()


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
        # Read from stdin if no file or text argument provided
        text = sys.stdin.read()

    # Apply humanization (structural rewriting)
    result = humanize(text)

    # Apply lexical simplification if enabled
    if args.lexical:
        try:
            dataset = HumanizeDataset.load(args.dataset)
        except FileNotFoundError:
            print(
                f"Error: Dataset file not found: {args.dataset}\n"
                "Please ensure the HumanizeDataset bundle file exists.\n"
                "You may need to create it from your existing parquet files.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError as e:
            print(
                f"Error: Invalid dataset file: {args.dataset}\n"
                f"Details: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        simplifier = LexicalSimplifier(dataset)
        result = simplifier.simplify(result)

    # Write output
    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(result)
        except IOError as e:
            print(f"Error writing to file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Print only the final output text to stdout
        print(result, end="")


if __name__ == "__main__":
    main()
