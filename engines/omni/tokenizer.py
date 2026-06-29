#!/usr/bin/env python3
"""Tokenize/detokenize subprocess for the omni engine.

Uses HuggingFace transformers for tokenization matching the official
Janus-Pro implementation.

Protocol:
  E:text\n         -> encode text, output space-separated ints
  D:id1 id2 ...\n  -> decode ids, output text
"""
import sys
from transformers import AutoTokenizer

MODEL = "/home/chenkailun.c/models/janus-pro-7b/"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith("E:"):
            text = line[2:]
            ids = tok.encode(text, add_special_tokens=True)
            print(" ".join(str(i) for i in ids), flush=True)
        elif line.startswith("D:"):
            ids_str = line[2:].strip()
            if not ids_str:
                continue
            ids = [int(x) for x in ids_str.split()]
            text = tok.decode(ids, skip_special_tokens=True)
            print(text, flush=True)

if __name__ == "__main__":
    main()