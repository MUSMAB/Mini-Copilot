"""Download the 10 dataset projects listed in data/dataset.json into data/raw/.

Each project is checked out at its pinned commit and only .py files are kept.
Usage: python scripts/download_dataset.py
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def fetch(project: dict) -> None:
    dest = RAW / project["name"].replace("/", "__")
    if dest.exists():
        print(f"skip {project['name']} (already downloaded)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        subprocess.run(["git", "-C", tmp, "fetch", "-q", "--depth", "1", project["url"], project["commit"]], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "-q", "FETCH_HEAD"], check=True)
        for py in Path(tmp).rglob("*.py"):
            if ".git" in py.parts:
                continue
            target = dest / py.relative_to(tmp)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py, target)
    print(f"done {project['name']}")


def main() -> None:
    projects = json.loads((ROOT / "data" / "dataset.json").read_text())["projects"]
    RAW.mkdir(parents=True, exist_ok=True)
    for project in projects:
        fetch(project)


if __name__ == "__main__":
    main()
