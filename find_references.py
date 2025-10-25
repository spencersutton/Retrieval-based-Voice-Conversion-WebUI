#!/usr/bin/env python3
"""
Find all references to top-level symbols from a Python file.

This script extracts all top-level symbols from a given Python file and searches
the current directory and subdirectories (excluding .venv) for references to those
symbols. It reports which symbols aren't used anywhere else in the codebase.

Usage:
    python find_references.py <file_path> [--workspace-root <root>] [--format FORMAT]

Formats:
    text     - Human-readable text output (default)
    json     - Machine-readable JSON output
    csv      - Comma-separated values
    problems - VS Code problem matcher format (file:line:col: warning: message)

Example:
    python find_references.py lib/utils.py
    python find_references.py src/models.py --workspace-root .
    python find_references.py tools/app.py --format problems --unused-only
"""

import argparse
import ast
import json
import os
import re
import sys
import warnings
from pathlib import Path

# Suppress SyntaxWarnings from the AST parser
warnings.filterwarnings("ignore", category=SyntaxWarning)


def extract_top_level_symbols(file_path: str) -> list[str]:
    """
    Extract all top-level symbols from a Python file.

    Top-level symbols include functions, classes, and module-level variables.

    Args:
        file_path: Path to the Python file

    Returns:
        List of symbol names
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError) as e:
        raise OSError(f"Cannot read file {file_path}: {e}")

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Cannot parse file {file_path}: {e}")

    symbols = []

    class TopLevelFinder(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            symbols.append(node.name)
            # Don't visit children - we only want top-level

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            symbols.append(node.name)
            # Don't visit children - we only want top-level

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            symbols.append(node.name)
            # Don't visit children - we only want top-level

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Skip private and magic variables
                    if not target.id.startswith("_"):
                        symbols.append(target.id)
            # Don't visit children

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name):
                if not node.target.id.startswith("_"):
                    symbols.append(node.target.id)
            # Don't visit children

    finder = TopLevelFinder()
    for node in ast.iter_child_nodes(tree):
        finder.visit(node)

    # Return unique symbols, preserving order
    seen = set()
    unique_symbols = []
    for sym in symbols:
        if sym not in seen:
            seen.add(sym)
            unique_symbols.append(sym)

    return unique_symbols


def _build_imports_map(
    python_files: list[str], target_file: str
) -> dict[str, set[str]]:
    """
    Build a mapping of files to symbols they import from the target file.

    Args:
        python_files: List of all Python files in the workspace
        target_file: The target file path to check imports from

    Returns:
        Dictionary mapping file paths to sets of symbol names imported from target_file
    """
    imports_map = {}
    target_module_name = _get_module_name(target_file)

    for py_file in python_files:
        imported_symbols = _extract_imports_from_file(py_file, target_module_name)
        if imported_symbols:
            imports_map[py_file] = imported_symbols

    return imports_map


def _get_module_name(file_path: str) -> str:
    """
    Convert a file path to a module name.

    Examples:
        /path/to/lib/utils.py -> lib.utils
        /path/to/tools/app.py -> tools.app

    Args:
        file_path: Path to the Python file

    Returns:
        Module name (dot-separated)
    """
    # Get the relative path without extension
    path = Path(file_path)
    parts = path.with_suffix("").parts

    # Find where the package/module structure starts
    # Look for common package indicators like 'src', 'lib', 'infer', 'tools', etc.
    # or the first part if absolute path contains directories

    module_parts = []
    found_package_root = False
    package_root_names = ("src", "lib", "infer", "tools", "app", "modules", "packages")

    for i, part in enumerate(parts):
        # Skip absolute path root on Unix (/), or drive letters on Windows (C:)
        if part == "" or (len(part) == 2 and part[1] == ":"):
            continue
        # Skip common workspace path components
        if part in ("home", "Users", "users"):
            continue

        # Check if this part is a package root
        is_package_root = part in package_root_names

        # Start collecting parts when we find a package root
        if is_package_root and not found_package_root:
            found_package_root = True

        # Include parts once we've found the package root
        if found_package_root:
            module_parts.append(part)

    if module_parts:
        return ".".join(module_parts)

    # Fallback: just use the last 2 parts or filename
    return ".".join(parts[-2:]) if len(parts) >= 2 else path.stem


def _extract_imports_from_file(file_path: str, target_module_name: str) -> set[str]:
    """
    Extract symbols imported from a target module in a file.

    Args:
        file_path: Path to the Python file to analyze
        target_module_name: The module name to search for (e.g., "lib.utils")

    Returns:
        Set of symbol names imported from the target module
    """
    imported_symbols = set()

    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return imported_symbols

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imported_symbols

    class ImportFinder(ast.NodeVisitor):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module is None:
                return

            module = node.module

            # Check if this import is from the target module (exact or parent match)
            if module == target_module_name or module.startswith(
                target_module_name + "."
            ):
                # Check for "from module import *"
                for alias in node.names:
                    if alias.name == "*":
                        # Import all - we can't determine specific symbols
                        # Add a special marker to indicate wildcard import
                        imported_symbols.add("*")
                    else:
                        imported_symbols.add(alias.name)

            # Also check if importing the target module as a submodule
            # e.g., "from infer.lib.train import utils" when target is "infer.lib.train.utils"
            elif target_module_name.startswith(module + "."):
                # The remaining part after the module should be what's imported
                remaining = target_module_name[len(module) + 1 :]
                # Check if any imported name matches the first component of remaining
                first_component = remaining.split(".")[0]
                for alias in node.names:
                    if alias.name == first_component:
                        # Track this as a module import
                        imported_name = alias.asname if alias.asname else alias.name
                        imported_symbols.add(f"_import_{imported_name}")

            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            # Check for "import module" style imports
            for alias in node.names:
                if alias.name == target_module_name or alias.name.startswith(
                    target_module_name + "."
                ):
                    # For "import module as m", symbols must be accessed as "m.symbol"
                    # We track the alias name as a pseudo-symbol
                    imported_symbols.add(f"_import_{alias.asname or alias.name}")

            self.generic_visit(node)

    finder = ImportFinder()
    finder.visit(tree)
    return imported_symbols


def find_all_symbol_references(
    target_file: str,
    workspace_root: str | None = None,
) -> dict[str, list[dict]]:
    """
    Find all references to top-level symbols from a target file.

    Args:
        target_file: Path to the Python file to extract symbols from
        workspace_root: Root directory to search (defaults to current dir)

    Returns:
        Dictionary mapping symbol names to lists of reference locations
    """
    if workspace_root is None:
        workspace_root = str(Path.cwd())

    target_file = str(Path(target_file).resolve())
    workspace_root = str(Path(workspace_root).resolve())

    # Verify file exists
    if not Path(target_file).exists():
        raise FileNotFoundError(f"File not found: {target_file}")

    # Extract top-level symbols
    symbols = extract_top_level_symbols(target_file)

    if not symbols:
        print(f"No top-level symbols found in {target_file}", file=sys.stderr)
        return {}

    print(
        f"Found {len(symbols)} top-level symbol(s): {', '.join(symbols)}",
        file=sys.stderr,
    )

    # Find all Python files in the workspace
    python_files = _find_python_files(workspace_root)

    # Build a mapping of file paths to imports from target file
    imports_map = _build_imports_map(python_files, target_file)

    # Search for references to each symbol
    symbol_references = {}
    for symbol in symbols:
        references = []
        for py_file in python_files:
            # Skip the target file itself for references (we'll add it separately)
            file_references = _find_references_in_file(py_file, symbol)

            # If this is not the target file, filter references based on imports
            if py_file != target_file:
                imported_symbols = imports_map.get(py_file, set())
                # Count if:
                # (1) symbol is explicitly imported, or
                # (2) wildcard import exists, or
                # (3) the module itself is imported (indicated by _import_ prefix)
                has_explicit_import = symbol in imported_symbols
                has_wildcard_import = "*" in imported_symbols
                has_module_import = any(
                    s.startswith("_import_") for s in imported_symbols
                )

                if has_explicit_import or has_wildcard_import or has_module_import:
                    references.extend(file_references)
            else:
                # For the target file itself, include all references
                references.extend(file_references)

        symbol_references[symbol] = references

    return symbol_references


def _find_python_files(
    root_dir: str, exclude_dirs: list[str] | None = None
) -> list[str]:
    """
    Find all Python files in the given directory.

    Args:
        root_dir: Root directory to search
        exclude_dirs: Directories to exclude from search

    Returns:
        List of Python file paths
    """
    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__", ".venv", "venv", ".env", "node_modules"]

    python_files = []

    for root, dirs, files in os.walk(root_dir):
        # Remove excluded directories from dirs in-place to prevent traversal
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


def _find_references_in_file(file_path: str, symbol_name: str) -> list[dict]:
    """
    Find all references to a symbol in a single file.

    Args:
        file_path: Path to the Python file
        symbol_name: Name of the symbol to find

    Returns:
        List of reference locations in this file
    """
    references = []

    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
            lines = source.split("\n")
    except (OSError, UnicodeDecodeError):
        return references

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # If AST parsing fails, fall back to regex-based search
        return _find_references_regex(file_path, symbol_name, lines)

    # Collect all definitions of the symbol
    definitions = _find_definitions_in_ast(tree, symbol_name)

    # Collect all references to the symbol
    references_in_tree = _find_references_in_ast(tree, symbol_name)

    # Convert AST nodes to location information
    for node in references_in_tree:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        if lineno is not None and col_offset is not None:
            ref_info = {
                "file": file_path,
                "line": lineno,
                "column": col_offset,
                "type": "reference",
            }
            references.append(ref_info)

    for node in definitions:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        if lineno is not None and col_offset is not None:
            ref_info = {
                "file": file_path,
                "line": lineno,
                "column": col_offset,
                "type": "definition",
            }
            references.append(ref_info)

    return references


def _find_references_regex(
    file_path: str, symbol_name: str, lines: list[str]
) -> list[dict]:
    """
    Find references using regex-based search (fallback for syntax errors).

    Args:
        file_path: Path to the Python file
        symbol_name: Name of the symbol to find
        lines: Lines of source code

    Returns:
        List of reference locations
    """
    references = []
    # Match whole words only
    pattern = r"\b" + re.escape(symbol_name) + r"\b"

    for line_num, line in enumerate(lines, 1):
        for match in re.finditer(pattern, line):
            ref_info = {
                "file": file_path,
                "line": line_num,
                "column": match.start(),
                "type": "reference",
            }
            references.append(ref_info)

    return references


def _find_definitions_in_ast(tree: ast.AST, symbol_name: str) -> list[ast.AST]:
    """
    Find all definitions of a symbol in an AST.

    Args:
        tree: AST tree
        symbol_name: Name of the symbol to find

    Returns:
        List of AST nodes that define the symbol
    """
    import ast

    definitions = []

    class DefinitionFinder(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name == symbol_name:
                definitions.append(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node.name == symbol_name:
                definitions.append(node)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node.name == symbol_name:
                definitions.append(node)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol_name:
                    definitions.append(target)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name) and node.target.id == symbol_name:
                definitions.append(node.target)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            if isinstance(node.target, ast.Name) and node.target.id == symbol_name:
                definitions.append(node.target)
            self.generic_visit(node)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            if isinstance(node.target, ast.Name) and node.target.id == symbol_name:
                definitions.append(node.target)
            self.generic_visit(node)

    finder = DefinitionFinder()
    finder.visit(tree)
    return definitions


def _find_references_in_ast(tree: ast.AST, symbol_name: str) -> list[ast.AST]:
    """
    Find all references to a symbol in an AST.

    Args:
        tree: AST tree
        symbol_name: Name of the symbol to find

    Returns:
        List of AST nodes that reference the symbol
    """
    import ast

    references = []

    class ReferenceFinder(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id == symbol_name and isinstance(node.ctx, ast.Load):
                references.append(node)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # Check if this is accessing the symbol as an attribute (e.g., module.symbol)
            if node.attr == symbol_name:
                references.append(node)
            self.generic_visit(node)

    finder = ReferenceFinder()
    finder.visit(tree)
    return references


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find unused top-level symbols in a Python file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find which top-level symbols in lib/utils.py are not used
  python find_references.py lib/utils.py
  
  # Use custom workspace root for searching
  python find_references.py src/models.py --workspace-root /path/to/project
  
  # Output as JSON
  python find_references.py lib/utils.py --format json
  
  # Output in VS Code problem matcher format
  python find_references.py tools/app.py --format problems --unused-only
        """,
    )

    parser.add_argument("file", help="Path to the Python file to analyze")
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="Root directory to search for references (defaults to current directory)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv", "problems"],
        default="text",
        help="Output format (default: text). 'problems' format is for VS Code problem matcher.",
    )
    parser.add_argument(
        "--unused-only",
        action="store_true",
        help="Only print unused symbols (hides used symbols)",
    )

    args = parser.parse_args()

    try:
        symbol_references = find_all_symbol_references(
            args.file,
            args.workspace_root,
        )

        # Identify unused symbols (those with no references except in the target file)
        target_file = str(Path(args.file).resolve())
        unused_symbols = []
        used_symbols = []

        for symbol, references in symbol_references.items():
            # Count internal references (excluding the definition itself)
            internal_refs = [
                ref
                for ref in references
                if ref["file"] == target_file and ref.get("type") != "definition"
            ]
            # Count external references
            external_refs = [ref for ref in references if ref["file"] != target_file]

            # A symbol is used if it has any internal or external references
            all_usage_refs = internal_refs + external_refs
            if not all_usage_refs:
                unused_symbols.append((symbol, references))
            else:
                used_symbols.append((symbol, all_usage_refs))

        # Format output
        if args.format == "json":
            if args.unused_only:
                output_data = {
                    "target_file": target_file,
                    "unused_symbols": [
                        {"symbol": sym, "references": refs}
                        for sym, refs in unused_symbols
                    ],
                }
            else:
                output_data = {
                    "target_file": target_file,
                    "unused_symbols": [
                        {"symbol": sym, "references": refs}
                        for sym, refs in unused_symbols
                    ],
                    "used_symbols": [
                        {"symbol": sym, "references": refs}
                        for sym, refs in used_symbols
                    ],
                }
            output = json.dumps(output_data, indent=2)
        elif args.format == "csv":
            lines = ["symbol,status,count"]
            for symbol, _ in unused_symbols:
                lines.append(f"{symbol},unused,0")
            if not args.unused_only:
                for symbol, refs in used_symbols:
                    lines.append(f"{symbol},used,{len(refs)}")
            output = "\n".join(lines)
        elif args.format == "problems":
            # Format for VS Code problem matcher
            # Each line: file:line:column: warning: message
            lines = []
            for symbol, references in unused_symbols:
                # Find the definition line
                for ref in references:
                    if ref.get("type") == "definition":
                        file_path = ref["file"]
                        line = ref["line"]
                        column = ref["column"]
                        lines.append(
                            f"{file_path}:{line}:{column}: warning: Unused symbol '{symbol}'"
                        )
                        break
            output = "\n".join(lines)
        else:  # text format
            lines = [f"Analysis of: {target_file}\n"]

            if unused_symbols:
                lines.append(f"❌ UNUSED SYMBOLS ({len(unused_symbols)}):")
                for symbol, references in unused_symbols:
                    lines.append(f"  • {symbol}")
                    if references:
                        # Show where it's defined
                        for ref in references:
                            if ref.get("type") == "definition":
                                lines.append(
                                    f"    Defined at: {ref['file']}:{ref['line']}:{ref['column']}"
                                )
            else:
                lines.append("✅ All symbols are used!")

            if not args.unused_only and used_symbols:
                lines.append(f"\n✅ USED SYMBOLS ({len(used_symbols)}):")
                for symbol, references in used_symbols:
                    lines.append(f"  • {symbol} ({len(references)} reference(s))")
                    # Group references by file
                    by_file = {}
                    for ref in references[:5]:  # Show first 5
                        file_path = ref["file"]
                        if file_path not in by_file:
                            by_file[file_path] = []
                        by_file[file_path].append(ref)

                    for file_path in sorted(by_file.keys()):
                        for ref in by_file[file_path]:
                            lines.append(
                                f"      {file_path}:{ref['line']}:{ref['column']}"
                            )

                    if len(references) > 5:
                        lines.append(f"      ... and {len(references) - 5} more")

            output = "\n".join(lines)

        print(output)

        print(
            f"\nFound {len(unused_symbols)} unused symbol(s), {len(used_symbols)} used symbol(s).",
            file=sys.stderr,
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
