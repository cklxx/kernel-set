"""Independent signature cross-check: C headers vs each binding's FFI decls.

Parses include/kernel_set/*.h to a canonical (name -> [arg C types], ret) map,
then compares against the declared FFI signatures extracted from each binding.
Reports any mismatch in argument count, type category, or return type.

Type categories used for comparison (so we compare semantics, not spelling):
  PTR    - any pointer (void*, const void*, T*, ks_stream_t, const T* const*)
  I32    - int / enum (ks_dtype_t, ks_status_t, ks_activation_t, ks_quant_mode_t,
           ks_memcpy_kind_t)  -- all 32-bit in the ABI
  I64    - int64_t
  U64    - uint64_t
  F32    - float
  SIZE   - size_t
  STR    - const char* return
  VOID   - void return / no-arg marker
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INCLUDE = os.path.join(REPO, "include", "kernel_set")

ENUM_TYPES = {
    "ks_status_t",
    "ks_dtype_t",
    "ks_activation_t",
    "ks_quant_mode_t",
    "ks_memcpy_kind_t",
}


def c_type_category(t: str) -> str:
    t = t.strip()
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t)
    if "*" in t:
        return "PTR"
    if t == "ks_stream_t":  # typedef void*
        return "PTR"
    if t in ("int64_t",):
        return "I64"
    if t in ("uint64_t",):
        return "U64"
    if t == "float":
        return "F32"
    if t == "size_t":
        return "SIZE"
    if t == "int":
        return "I32"
    if t in ENUM_TYPES:
        return "I32"
    if t == "void":
        return "VOID"
    raise ValueError(f"unknown C type: {t!r}")


def ret_category(t: str) -> str:
    t = t.strip()
    if t == "ks_status_t":
        return "I32"  # status return
    if t == "const char*" or t.endswith("char*"):
        return "STR"
    if t == "int":
        return "I32"
    raise ValueError(f"unknown return type: {t!r}")


def parse_headers() -> dict:
    """Return {func_name: {'args': [cat,...], 'ret': cat, 'raw': str}}."""
    text = ""
    for fn in sorted(os.listdir(INCLUDE)):
        if fn.endswith(".h"):
            with open(os.path.join(INCLUDE, fn)) as f:
                # Drop preprocessor directive lines (#define KS_API ... etc.)
                for line in f:
                    if line.lstrip().startswith("#"):
                        continue
                    text += line
            text += "\n"
    # Collapse to find each KS_API ... ;
    flat = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)  # strip block comments
    flat = re.sub(r"\s+", " ", flat)
    decls = re.findall(r"KS_API\s+(.*?);", flat)
    out = {}
    for d in decls:
        m = re.match(r"(.+?)\b(ks_[a-z0-9_]+)\s*\((.*)\)\s*$", d)
        if not m:
            raise ValueError(f"could not parse decl: {d!r}")
        ret_raw, name, args_raw = m.group(1), m.group(2), m.group(3)
        ret = ret_category(ret_raw.strip())
        args_raw = args_raw.strip()
        if args_raw in ("void", ""):
            arg_cats = []
        else:
            # split top-level commas (no nested parens in these decls)
            parts = [p.strip() for p in args_raw.split(",")]
            arg_cats = []
            for p in parts:
                # remove the parameter name: take everything before the last
                # identifier (which is the name), unless it's purely a type.
                # Strip a trailing identifier if present.
                p2 = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*$", "", p).strip()
                # If stripping removed the whole type (e.g. param was just a
                # type like "int"), fall back to the original.
                type_str = p2 if p2 else p
                # handle "const T* const*" etc -> still PTR via '*'
                arg_cats.append(c_type_category(type_str))
        out[name] = {"args": arg_cats, "ret": ret, "raw": d.strip()}
    return out


# --------------------------------------------------------------------------
# Python ctypes extraction
# --------------------------------------------------------------------------

PY_CTYPE_CAT = {
    "c_void_p": "PTR",
    "_VP": "PTR",
    "_STREAM": "PTR",
    "c_char_p": "STR",
    "c_int": "I32",
    "_DTYPE": "I32",
    "_STATUS": "I32",
    "c_int32": "I32",
    "c_int64": "I64",
    "c_uint64": "U64",
    "c_float": "F32",
    "c_size_t": "SIZE",
}


def py_arg_cat(tok: str) -> str:
    tok = tok.strip()
    if tok.startswith("POINTER("):
        return "PTR"
    if tok in PY_CTYPE_CAT:
        return PY_CTYPE_CAT[tok]
    raise ValueError(f"unknown python ctype token: {tok!r}")


def py_ret_cat(tok: str) -> str:
    tok = tok.strip()
    if tok in ("_STATUS", "c_int", "_DTYPE", "_STATUS"):
        return "I32"
    if tok == "c_char_p":
        return "STR"
    raise ValueError(f"unknown python return token: {tok!r}")


def parse_python() -> dict:
    path = os.path.join(REPO, "bindings", "python", "kernel_set", "_lib.py")
    with open(path) as f:
        src = f.read()
    # Find each _decl("name", [ ... ] (, restype)? )
    out = {}
    # Use a tokenizer over balanced parens for each _decl call.
    i = 0
    while True:
        m = re.search(r'_decl\(\s*"(ks_[a-z0-9_]+)"\s*,', src[i:])
        if not m:
            break
        name = m.group(1)
        start = i + m.end()  # position right after the comma
        # Now parse the argtypes list: find the matching [...]
        # Skip whitespace
        j = start
        while src[j] in " \n\t":
            j += 1
        assert src[j] == "[", f"expected [ for {name}, got {src[j]!r}"
        depth = 0
        k = j
        while k < len(src):
            if src[k] == "[":
                depth += 1
            elif src[k] == "]":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        list_body = src[j + 1 : k]
        # strip python inline comments (# ... to end of line) inside the list
        list_body = re.sub(r"#[^\n]*", "", list_body)
        # restype: text after the list up to the closing ) of _decl
        rest = src[k + 1 :]
        # the _decl call closes with ')'; find restype if present
        rm = re.match(r"\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", rest)
        if rm:
            ret = py_ret_cat(rm.group(1))
        else:
            ret = "I32"  # default restype=_STATUS
        # tokenize list_body by top-level commas
        toks = split_top_commas(list_body)
        arg_cats = [py_arg_cat(t) for t in toks if t.strip()]
        out[name] = {"args": arg_cats, "ret": ret}
        i = k
    return out


def split_top_commas(s: str) -> list:
    parts = []
    depth = 0
    cur = ""
    for ch in s:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def compare(name_lang: str, headers: dict, binding: dict) -> int:
    mismatches = 0
    hnames = set(headers)
    bnames = set(binding)
    missing = hnames - bnames
    extra = bnames - hnames
    if missing:
        print(f"[{name_lang}] MISSING decls: {sorted(missing)}")
        mismatches += len(missing)
    if extra:
        print(f"[{name_lang}] EXTRA decls: {sorted(extra)}")
        mismatches += len(extra)
    for fn in sorted(hnames & bnames):
        h = headers[fn]
        b = binding[fn]
        if h["args"] != b["args"]:
            print(f"[{name_lang}] ARG MISMATCH {fn}:")
            print(f"    header : {h['args']}")
            print(f"    binding: {b['args']}")
            mismatches += 1
        if h["ret"] != b["ret"]:
            print(f"[{name_lang}] RET MISMATCH {fn}: header={h['ret']} binding={b['ret']}")
            mismatches += 1
    return mismatches


# --------------------------------------------------------------------------
# Rust sys.rs extraction
# --------------------------------------------------------------------------


def rust_arg_cat(ty: str) -> str:
    ty = ty.strip()
    if "*" in ty:  # *mut / *const pointers (incl. *const *const)
        return "PTR"
    if ty == "ks_stream_t":  # type alias for *mut c_void
        return "PTR"
    if ty in ("i64",):
        return "I64"
    if ty in ("u64",):
        return "U64"
    if ty in ("f32",):
        return "F32"
    if ty in ("usize",):  # size_t
        return "SIZE"
    if ty in ("c_int", "i32"):
        return "I32"
    if ty.startswith("ks_"):  # repr(C) enum -> int-sized
        return "I32"
    raise ValueError(f"unknown rust type: {ty!r}")


def rust_ret_cat(ty: str) -> str:
    ty = ty.strip()
    if ty == "ks_status_t":
        return "I32"
    if ty in ("c_int", "i32"):
        return "I32"
    if "*const c_char" in ty or "c_char" in ty:
        return "STR"
    raise ValueError(f"unknown rust return type: {ty!r}")


def parse_rust() -> dict:
    path = os.path.join(REPO, "bindings", "rust", "src", "sys.rs")
    with open(path) as f:
        src = f.read()
    out = {}
    # Match `pub fn ks_xxx( ... ) -> RET ;` (balanced parens by regex on the
    # known-simple decls: no nested parens inside arg lists).
    for m in re.finditer(
        r"pub\s+fn\s+(ks_[a-z0-9_]+)\s*\((.*?)\)\s*->\s*([^;{]+);",
        src,
        flags=re.S,
    ):
        name = m.group(1)
        args_raw = m.group(2).strip()
        ret_raw = m.group(3).strip()
        ret = rust_ret_cat(ret_raw)
        if not args_raw:
            arg_cats = []
        else:
            parts = split_top_commas(args_raw)
            arg_cats = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # form: `name: type`
                colon = p.index(":")
                ty = p[colon + 1 :].strip()
                arg_cats.append(rust_arg_cat(ty))
        out[name] = {"args": arg_cats, "ret": ret}
    return out


# --------------------------------------------------------------------------
# TypeScript (koffi C-prototype strings in lib.ts) extraction
# --------------------------------------------------------------------------

TS_ALIAS_CAT = {
    "ks_devptr": "PTR",  # alias for void*
    "ks_stream_t": "PTR",  # alias for void*
    "ks_status_t": "I32",  # alias for int
    "ks_dtype_t": "I32",
    "ks_activation_t": "I32",
    "ks_quant_mode_t": "I32",
    "ks_memcpy_kind_t": "I32",
}


def ts_arg_cat(ty: str) -> str:
    ty = ty.strip()
    if "*" in ty:  # raw pointer spellings (e.g. `int *`, `void *`)
        return "PTR"
    if ty in TS_ALIAS_CAT:
        return TS_ALIAS_CAT[ty]
    if ty == "int64":
        return "I64"
    if ty == "uint64":
        return "U64"
    if ty == "float":
        return "F32"
    if ty == "size_t":
        return "SIZE"
    if ty == "int":
        return "I32"
    raise ValueError(f"unknown ts type: {ty!r}")


def ts_ret_cat(ty: str) -> str:
    ty = ty.strip()
    if ty in TS_ALIAS_CAT:
        return TS_ALIAS_CAT[ty]
    if ty == "int":
        return "I32"
    if "char" in ty:  # `const char *`
        return "STR"
    raise ValueError(f"unknown ts return type: {ty!r}")


def parse_ts() -> dict:
    path = os.path.join(REPO, "bindings", "ts", "src", "lib.ts")
    with open(path) as f:
        src = f.read()
    out = {}
    # Each decl is L.func('<C-proto>'). Grab the proto string contents.
    for proto in re.findall(r"L\.func\(\s*'([^']+)'", src):
        # proto looks like: "RET ks_name(arg1, arg2, ...)"
        m = re.match(r"\s*(.+?)\b(ks_[a-z0-9_]+)\s*\((.*)\)\s*$", proto)
        if not m:
            raise ValueError(f"cannot parse ts proto: {proto!r}")
        ret_raw, name, args_raw = m.group(1), m.group(2), m.group(3)
        ret = ts_ret_cat(ret_raw.strip())
        args_raw = args_raw.strip()
        if args_raw in ("void", ""):
            arg_cats = []
        else:
            parts = [p.strip() for p in args_raw.split(",")]
            arg_cats = []
            for p in parts:
                # strip the _Out_/_In_ koffi direction markers
                p = re.sub(r"\b_(Out|In|Inout)_\b", "", p).strip()
                # form "type name" (or "type * name"); strip trailing identifier
                p2 = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*$", "", p).strip()
                type_str = p2 if p2 else p
                arg_cats.append(ts_arg_cat(type_str))
        out[name] = {"args": arg_cats, "ret": ret}
    return out


def main() -> int:
    headers = parse_headers()
    print(f"[xcheck] parsed {len(headers)} functions from headers")
    grand = 0

    # Python
    py = parse_python()
    print(f"[xcheck] parsed {len(py)} python decls")
    n = compare("python", headers, py)
    print(
        "[xcheck] python: ALL SIGNATURES MATCH"
        if n == 0
        else f"[xcheck] python: {n} mismatch(es)"
    )
    grand += n

    # Rust
    rs = parse_rust()
    print(f"[xcheck] parsed {len(rs)} rust extern decls")
    n = compare("rust", headers, rs)
    print(
        "[xcheck] rust: ALL SIGNATURES MATCH"
        if n == 0
        else f"[xcheck] rust: {n} mismatch(es)"
    )
    grand += n

    # TypeScript
    ts = parse_ts()
    print(f"[xcheck] parsed {len(ts)} typescript decls")
    n = compare("ts", headers, ts)
    print(
        "[xcheck] ts: ALL SIGNATURES MATCH"
        if n == 0
        else f"[xcheck] ts: {n} mismatch(es)"
    )
    grand += n

    # Go: signatures are validated by cgo at compile time (the C compiler checks
    # every C.ks_* call against the real header). Not re-parsed here.
    print("[xcheck] go: signatures validated by cgo compile (see build step)")

    return 0 if grand == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
