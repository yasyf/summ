def dedent(text: str) -> str:
    return "\n".join(map(str.strip, text.splitlines())).strip()
