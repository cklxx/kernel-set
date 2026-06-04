from pathlib import Path

from megakernels.demos.latency.mk import LatencyMK_Interpreter
from megakernels.demos.latency.python_vm import (
    INSTRUCTION_TO_SOLVER as LATENCY_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.latency.scheduler import LatencyScheduleBuilder
from megakernels.demos.throughput.mk import ThroughputMK_Interpreter
from megakernels.demos.throughput.python_vm import (
    INSTRUCTION_TO_SOLVER as THROUGHPUT_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.throughput.scheduler import ThroughputScheduleBuilder
from megakernels.mk import MK_Interpreter
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import ScheduleBuilder

BUILDER_MAP = {
    "latency": LatencyScheduleBuilder,
    "throughput": ThroughputScheduleBuilder,
}

MK_INTERPRETER_MAP = {
    "latency": LatencyMK_Interpreter,
    "throughput": ThroughputMK_Interpreter,
}

INSTRUCTION_TO_SOLVER_MAP = {
    "latency": LATENCY_INSTRUCTION_TO_SOLVER,
    "throughput": THROUGHPUT_INSTRUCTION_TO_SOLVER,
}


def make_schedule_builder(mode: str) -> ScheduleBuilder:
    return BUILDER_MAP[mode]()


def make_mk_interpreter(mode: str, mk_dir: Path) -> MK_Interpreter:
    return MK_INTERPRETER_MAP[mode](mk_dir)


def make_pyvm_interpreter(mode: str) -> PyVM_Interpreter:
    return PyVM_Interpreter(INSTRUCTION_TO_SOLVER_MAP[mode])
