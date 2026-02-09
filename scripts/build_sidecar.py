#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = ROOT / "binaries"
ENTRYPOINT = ROOT / "hy3dgen" / "sidecar" / "main.py"


def target_name() -> str:
    base = "python-backend"
    if platform.system().lower().startswith("win"):
        return f"{base}.exe"
    return base


def target_triple_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if machine in {"amd64", "x86_64"}:
        arch = "x86_64"
    elif machine in {"aarch64", "arm64"}:
        arch = "aarch64"
    else:
        arch = machine

    if system.startswith("linux"):
        return f"python-backend-{arch}-unknown-linux-gnu"
    if system.startswith("win"):
        return f"python-backend-{arch}-pc-windows-msvc.exe"

    # Fallback for unsupported OS in this project baseline.
    return target_name()


def build(dry_run: bool = False) -> int:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    pyinstaller = shutil.which("pyinstaller")
    if not pyinstaller:
        if dry_run:
            print("pyinstaller not found (dry-run mode).")
            print("Install with: pip install pyinstaller")
            return 0
        print("pyinstaller not found. Install with: pip install pyinstaller", file=sys.stderr)
        return 2

    cmd = [
        pyinstaller,
        "--onefile",
        "--clean",
        "--name",
        "python-backend",
        "--distpath",
        str(BIN_DIR),
        "--workpath",
        str(ROOT / "build" / "pyinstaller"),
        "--specpath",
        str(ROOT / "build" / "pyinstaller"),
        str(ENTRYPOINT),
    ]

    print("Running:", " ".join(cmd))
    if dry_run:
        return 0

    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    if proc.returncode != 0:
        return proc.returncode

    expected = BIN_DIR / target_name()
    if expected.exists():
        triple_path = BIN_DIR / target_triple_name()
        shutil.copy2(expected, triple_path)
        print(f"Built sidecar: {expected}")
        print(f"Copied sidecar for Tauri externalBin: {triple_path}")
        return 0

    print("PyInstaller finished but target binary was not found", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Python sidecar for Tauri externalBin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return build(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
