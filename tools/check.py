#!/usr/bin/env python3
"""Dependency-free pre-flight checks for this repository.

    python tools/check.py

It byte-compiles every module (catches syntax errors without importing torch),
asserts the two behavioural fixes are still present, and refuses to pass if
compiled artefacts have been committed.
"""

from __future__ import annotations

import pathlib
import py_compile
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = [ROOT / name for name in ("data", "models", "options", "util", "train.py", "test.py")]

GUARDS = [
    ("data/unaligned_dataset.py", "has_depth", "C 域（几何图）目录缺失时应自动跳过"),
    ("models/cycle_gan_model.py", "self.clip_model.encode_image", "语义一致损失应复用已加载的 CLIP"),
    ("models/cycle_gan_model.py", "netGeom is not None", "几何网络权重缺失时应跳过该约束"),
]


def python_files():
    for target in TARGETS:
        if target.is_dir():
            yield from sorted(target.rglob("*.py"))
        elif target.exists():
            yield target


def main() -> int:
    problems: list[str] = []
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path in python_files():
            checked += 1
            try:
                py_compile.compile(str(path), cfile=str(pathlib.Path(tmp) / "x.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                problems.append(f"syntax: {path.relative_to(ROOT)} -> {exc.msg.strip()}")

    for rel, needle, why in GUARDS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        if needle not in text:
            problems.append(f"guard: {rel} 缺少 {needle!r}（{why}）")

    junk = [p for p in ROOT.rglob("*.pyc")] + [p for p in ROOT.rglob("__pycache__")]
    junk = [p for p in junk if ".git" not in p.parts]
    if junk:
        problems.append(f"artefacts: {len(junk)} 个 .pyc / __pycache__ 被提交进了仓库")

    if problems:
        print("FAILED")
        for p in problems:
            print("  -", p)
        return 1

    print(f"OK — {checked} modules byte-compiled, {len(GUARDS)} guards present, no artefacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
