from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def find_repo_root(start: Optional[Path] = None) -> Path:
    start_path = start or Path.cwd()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start_path),
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            root = res.stdout.strip()
            if root:
                return Path(root)
    except Exception:
        pass

    cur = start_path.resolve()
    while True:
        if (cur / ".git").exists() or (cur / "config" / "settings.yaml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start_path.resolve()


def ensure_repo_root(start: Optional[Path] = None) -> Path:
    root = find_repo_root(start)
    os.chdir(root)
    return root
