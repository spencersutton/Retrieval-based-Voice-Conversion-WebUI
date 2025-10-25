When generating type annotations use the new typing syntax (e.g. list[int], dict[str, str]). When a type cannot be determined, use 'object' as the type. Don't use generic types without type parameters (e.g. use 'list[int]' instead of 'list').

When working with Python packages always use uv. For example use uv run main.py instead of python main.py.

When installing or updating packages always use uv pip. For example use uv pip install package-name instead of pip install package-name.

When working with paths, always use pathlib.Path for path manipulations instead of string operations.