from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_status import all_status_as_dicts, status_summary_counts


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the current keep/modify/legacy classification for the rebuilt project."
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "summary_counts": status_summary_counts(),
        "components": all_status_as_dicts(),
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
        return

    print("Project rebuild status")
    print(json.dumps(payload["summary_counts"], indent=2, ensure_ascii=False))
    print("")
    for item in payload["components"]:
        print(f"[{item['category']}] {item['path']}")
        print(f"  summary: {item['summary']}")
        print(f"  next:    {item['next_action']}")


if __name__ == "__main__":
    main()
