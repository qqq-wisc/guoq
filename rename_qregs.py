#!/usr/bin/env python3
"""Rewrite QASM files so they have a single qreg named "q".

Multiple qregs are concatenated in declaration order (each reg gets a base
offset), and every reference name[i] is remapped to q[offset + i]. Files that
already have exactly one qreg named "q" are left untouched.

Usage: python3 rename_qregs.py [--dry-run] dir_or_file [...]
"""

import argparse
import re
import sys
from pathlib import Path

QREG_RE = re.compile(r"^\s*qreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*;")


def rewrite(text):
    """Return rewritten text, or None if no change is needed."""
    lines = text.splitlines(keepends=True)

    # Collect qreg declarations in order.
    regs = []  # (line_index, name, size)
    for i, line in enumerate(lines):
        m = QREG_RE.match(line)
        if m:
            regs.append((i, m.group(1), int(m.group(2))))

    if not regs:
        return None
    if len(regs) == 1 and regs[0][1] == "q":
        return None

    # Assign each reg a base offset in the merged register.
    offsets = {}
    total = 0
    for _, name, size in regs:
        if name in offsets:
            raise ValueError(f"duplicate qreg name '{name}'")
        offsets[name] = total
        total += size

    # Replace the first qreg declaration with the merged one, drop the rest.
    decl_lines = {i for i, _, _ in regs}
    first_decl = regs[0][0]
    out = []
    ref_re = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in offsets) + r")\s*\[\s*(\d+)\s*\]"
    )

    def repl(m):
        return f"q[{offsets[m.group(1)] + int(m.group(2))}]"

    for i, line in enumerate(lines):
        if i == first_decl:
            out.append(f"qreg q[{total}];\n")
        elif i in decl_lines:
            continue
        else:
            out.append(ref_re.sub(repl, line))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.paths):
        if p.is_dir():
            files.extend(sorted(p.glob("*.qasm")))
        else:
            files.append(p)

    changed = 0
    for f in files:
        new = rewrite(f.read_text())
        if new is None:
            continue
        changed += 1
        print(("would rewrite" if args.dry_run else "rewrote") + f" {f}")
        if not args.dry_run:
            f.write_text(new)
    print(f"{changed} file(s) changed, {len(files) - changed} untouched")


if __name__ == "__main__":
    main()
