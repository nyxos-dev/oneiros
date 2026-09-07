#!/bin/bash
# Regenerate the Oneiros training corpus from the NyxOS source tree.
# NyxOS-authored C only: kernel/** + top-level user/*. Excludes Fable's lang/
# toolchain and vendored code. The source is read ONLY as training data.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"                 # .../nyx-ai/data
NYX="${1:-$(cd "$HERE/../.." && pwd)/nyx-os}"          # sibling nyx-os repo
cd "$NYX"

{ find kernel -type f \( -name '*.c' -o -name '*.h' \)
  find user -maxdepth 1 -type f \( -name '*.c' -o -name '*.h' \)
} | sort | xargs cat > "$HERE/corpus.txt"

# Code-logic corpus: drop hex data-table lines (font bitmaps / embedded assets /
# KAT vectors) which are ~85% of the raw bytes and swamp a small model.
grep -avE '0x[0-9A-Fa-f]{2},' "$HERE/corpus.txt" > "$HERE/corpus_clean.txt"

echo "corpus.txt:       $(wc -c < "$HERE/corpus.txt") bytes"
echo "corpus_clean.txt: $(wc -c < "$HERE/corpus_clean.txt") bytes"
