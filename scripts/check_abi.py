#!/usr/bin/env python3
"""Verify every C ABI function declared in include/kernel_set/*.h has a
definition somewhere under kernels/src/*.cu.

Usage: python3 scripts/check_abi.py
Exit code 0 if every declared `ks_*` symbol is defined, 1 otherwise.
This is an orchestrator integration check, independent of the unit tests.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INC = os.path.join(ROOT, "include", "kernel_set")
SRC = os.path.join(ROOT, "kernels", "src")

# Declarations look like: KS_API ks_status_t ks_foo(...);  (also const char*, int)
DECL_RE = re.compile(r"\bKS_API\b[^;{]*?\b(ks_[a-z0-9_]+)\s*\(")
# A definition is `ks_foo(` that is followed (eventually) by a `{` not `;`.
DEFN_RE = re.compile(r"\b(ks_[a-z0-9_]+)\s*\([^;]*?\)\s*\{", re.S)


def read(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()


def collect(glob_dir, exts):
    out = []
    for dirpath, _, files in os.walk(glob_dir):
        for fn in files:
            if any(fn.endswith(e) for e in exts):
                out.append(os.path.join(dirpath, fn))
    return out


def main():
    declared = set()
    for h in collect(INC, (".h",)):
        for m in DECL_RE.finditer(read(h)):
            declared.add(m.group(1))

    defined = set()
    for c in collect(SRC, (".cu", ".cpp", ".cc")):
        text = read(c)
        for m in DEFN_RE.finditer(text):
            defined.add(m.group(1))

    missing = sorted(declared - defined)
    extra = sorted(defined - declared - {"main"})

    print(f"declared (include/): {len(declared)}")
    print(f"defined   (kernels/src/): {len(defined & declared)} of declared")
    if missing:
        print(f"\nMISSING DEFINITIONS ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
    if extra:
        print(f"\nDefined but not declared ({len(extra)}): {', '.join(extra)}")

    if missing:
        print("\nFAIL: some declared ABI functions are not implemented.")
        return 1
    print("\nOK: every declared ABI function has a definition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
