#!/usr/bin/env python3
"""Clean up checkpoint directory by removing large intermediate files.

Deletes:
1. Files named exactly 'data.pt'
2. Files matching pattern 'model_world_size_*\.pt'

Usage:
    python clean_checkpoints.py [root_dir]

If root_dir is not provided, defaults to
'checkpoints_verl_mt/lic_mixed' relative to script location.
"""

import re
import sys
from pathlib import Path
from typing import Iterable


def iter_target_files(root: Path) -> Iterable[Path]:
    """Yield files under root matching deletion criteria."""
    pattern = re.compile(r"model_world_size_.*\.pt$")  # Match any characters after prefix
    for path in root.rglob("*"):
        if path.is_file():
            if path.name == "data.pt" or pattern.match(path.name):
                yield path


def delete_files(files: Iterable[Path]) -> None:
    deleted = 0
    for file in files:
        try:
            file.unlink()
            print(f"Deleted {file}")
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {file}: {e}")
    print(f"Total deleted: {deleted}")


def main() -> None:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).expanduser().resolve()
    else:
        root = Path(__file__).parent / "checkpoints/lic_mixed"

    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    files_to_delete = list(iter_target_files(root))
    if not files_to_delete:
        print("No matching files found.")
        return

    print(f"Found {len(files_to_delete)} files to delete under {root}")
    for file in files_to_delete:
        print(file)
    delete_files(files_to_delete)


if __name__ == "__main__":
    main()
