import sys
from pathlib import Path


def get_mk_func(mk_dir: Path):
    sys.path.append(str(mk_dir.expanduser().absolute()))
    from mk_llama import mk_llama  # type: ignore

    return mk_llama


class MK_Interpreter:
    def __init__(self, mk_dir: Path):
        self.mk_func = get_mk_func(mk_dir)

    def interpret(self, globs):
        raise NotImplementedError
