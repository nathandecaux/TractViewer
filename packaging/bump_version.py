"""Print or change the version (tractviewer/_version.py, single source).

    python packaging/bump_version.py              -> print the current version
    python packaging/bump_version.py patch        -> 0.1.0 -> 0.1.1
    python packaging/bump_version.py minor|major
    python packaging/bump_version.py 1.4.0        -> explicit version
Prints the new version on stdout.
"""
import re
import sys
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "tractviewer" / "_version.py"
PATTERN = re.compile(r'__version__ = "(\d+)\.(\d+)\.(\d+)"')


def main(argv):
    text = FILE.read_text()
    m = PATTERN.search(text)
    if not m:
        raise SystemExit(f"no __version__ = \"X.Y.Z\" in {FILE}")
    major, minor, patch = map(int, m.groups())
    if not argv:
        print(f"{major}.{minor}.{patch}")
        return
    arg = argv[0].lstrip("v")
    if arg == "major":
        new = f"{major + 1}.0.0"
    elif arg == "minor":
        new = f"{major}.{minor + 1}.0"
    elif arg == "patch":
        new = f"{major}.{minor}.{patch + 1}"
    elif re.fullmatch(r"\d+\.\d+\.\d+", arg):
        new = arg
    else:
        raise SystemExit(f"expected patch, minor, major or X.Y.Z, got {argv[0]!r}")
    FILE.write_text(PATTERN.sub(f'__version__ = "{new}"', text, count=1))
    print(new)


if __name__ == "__main__":
    main(sys.argv[1:])
