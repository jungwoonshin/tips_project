"""Append-only JSONL history of AutoResearch iterations."""

from __future__ import annotations

import json
import time
from pathlib import Path


class History:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        self._cache.append(json.loads(line))
                    except Exception:
                        pass

    def append(self, entry: dict) -> None:
        entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
        self._cache.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def tail(self, n: int) -> list[dict]:
        return list(self._cache[-n:])

    def all(self) -> list[dict]:
        return list(self._cache)
