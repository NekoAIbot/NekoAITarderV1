#!/usr/bin/env python3
"""Fail CI if tracked binary-like artifacts are present in git history snapshot."""
import subprocess
import sys

BLOCKED_EXTENSIONS = (".joblib", ".pkl", ".h5", ".keras", ".pyc", ".db")


def main() -> int:
    files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    bad = [f for f in files if f.endswith(BLOCKED_EXTENSIONS) or "__pycache__/" in f]

    if bad:
        print("❌ Tracked binary-like artifacts found:")
        for f in bad:
            print(f" - {f}")
        return 1

    print("✅ No tracked binary-like artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
