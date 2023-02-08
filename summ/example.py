import importlib.util
import sys
from pathlib import Path


def import_examples():
    path = (
        Path(__file__).parent.parent
        / "examples"
        / "otter"
        / "implementation"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("examples", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["examples"] = module
    spec.loader.exec_module(module)
    return module


def main():
    module = import_examples()
    module.main()


if __name__ == "__main__":
    main()
