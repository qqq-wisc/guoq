#!/usr/bin/env bash
# Fetch the full upstream benchmark corpus (~409 MB) into benchmarks-full/.
#
# The repo vendors only a small curated subset in benchmarks/, enough for the corpus
# tests. Differential runs against the reference implementation want the whole set.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/benchmarks-full"

if [ -d "$DEST" ]; then
    echo "$DEST already exists; remove it to re-fetch." >&2
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Cloning upstream benchmarks..."
git clone --depth=1 --filter=blob:none --sparse \
    https://github.com/qqq-wisc/guoq.git "$TMP/guoq"
git -C "$TMP/guoq" sparse-checkout set benchmarks

mv "$TMP/guoq/benchmarks" "$DEST"
echo "Benchmarks available at $DEST"
