#!/usr/bin/env python3
"""
Script to delete lines referenced in ruff ERA001 (deprecated-calls) warnings.
Run: ruff check --select ERA001 | grep -- '-->' | python delete_era001_lines.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_ruff_output(lines):
    """Parse ruff output and extract file:line references."""
    file_lines = defaultdict(list)

    for line in lines:
        line = line.strip()
        if not line or "-->" not in line:
            continue

        # Parse format: "  --> path/to/file.py:line_number:column"
        match = re.search(r"-->\s+(.+?):(\d+):", line)
        if match:
            filepath = match.group(1)
            line_num = int(match.group(2))
            file_lines[filepath].append(line_num)

    return file_lines


def delete_lines_from_file(filepath, line_numbers):
    """Delete specified lines from a file."""
    try:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠ File not found: {filepath}", file=sys.stderr)
            return False

        # Sort line numbers in descending order to avoid index shifting
        line_numbers = sorted(set(line_numbers), reverse=True)

        # Read file
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        # Delete lines (converting 1-indexed to 0-indexed)
        for line_num in line_numbers:
            if 0 < line_num <= len(lines):
                del lines[line_num - 1]
            else:
                print(f"⚠ Line {line_num} out of range for {filepath}", file=sys.stderr)

        # Write file back
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        print(
            f"✓ {filepath}: deleted {len(line_numbers)} line(s) {sorted(line_numbers, reverse=False)}"
        )
        return True
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    # Read from stdin
    input_lines = sys.stdin.readlines()

    if not input_lines:
        print(
            "No input provided. Usage: ruff check --select ERA001 | grep -- '-->' | python delete_era001_lines.py"
        )
        sys.exit(1)

    # Parse the ruff output
    file_lines = parse_ruff_output(input_lines)

    if not file_lines:
        print("No lines to delete found in ruff output.")
        sys.exit(0)

    print(f"Found {len(file_lines)} file(s) to process:")

    # Process each file
    total_deleted = 0
    for filepath in sorted(file_lines.keys()):
        line_numbers = file_lines[filepath]
        if delete_lines_from_file(filepath, line_numbers):
            total_deleted += len(line_numbers)

    print(f"\nTotal: {total_deleted} line(s) deleted")


if __name__ == "__main__":
    main()
