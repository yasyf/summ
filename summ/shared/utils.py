def dedent(text: str) -> str:
    """A more lenient version of `textwrap.dedent`."""

    return "\n".join(map(str.strip, text.splitlines())).strip()
