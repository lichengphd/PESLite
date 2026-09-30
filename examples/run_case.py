"""Run a case from the command line.

    python examples/run_case.py gfl-example
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peslite import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
