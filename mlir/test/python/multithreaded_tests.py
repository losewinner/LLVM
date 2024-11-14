import concurrent.futures
import functools
import gc
import importlib.util
import sys
import threading
import tempfile

from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import pytest

import mlir.dialects.arith as arith
from mlir.ir import Context, Location, Module, IntegerType, F64Type, InsertionPoint



def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def copy_and_update(src_filepath: Path, dst_filepath: Path):
    # We should remove all calls like `run(testMethod)`
    with open(src_filepath, "r") as reader, open(dst_filepath, "w") as writer:
        while True:
            src_line = reader.readline()
            if len(src_line) == 0:
                break
            skip_lines = [
                "run(",
                "@run",
                "@constructAndPrintInModule",
            ]
            if any(src_line.startswith(line) for line in skip_lines):
                continue
            writer.write(src_line)


def run(f):
    f()


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)


def run_with_context_and_location(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


test_modules = [
    ("execution_engine", run),  # Fail
    ("pass_manager", run),  # Fail

    # Dialects tests
    ("dialects/affine", constructAndPrintInModule),  # Fail
    ("dialects/vector", run_with_context_and_location),  # Fail

    # IR tests
    ("ir/affine_expr", run),  # Pass
    ("ir/affine_map", run),  # Pass
    ("ir/array_attributes", run),  # Pass
    ("ir/attributes", run),  # Pass
    ("ir/blocks", run),  # Pass
    ("ir/builtin_types", run),  # Pass
    ("ir/context_managers", run),  # Pass
    ("ir/debug", run),  # Fail
    ("ir/diagnostic_handler", run),  # Fail
    ("ir/dialects", run),  # Fail
    ("ir/exception", run),  # Fail
    ("ir/insertion_point", run),  # Pass
    ("ir/insertion_point", run),  # Pass
    ("ir/integer_set", run),  # Pass
    ("ir/location", run),  # Pass
    ("ir/module", run),  # Pass but may fail randomly on mlirOperationDump in testParseSuccess
    ("ir/operation", run),  # Pass
    ("ir/symbol_table", run),  # Pass
    ("ir/value", run),  # Fail/Crash

]


def add_existing_tests(test_prefix: str = "_original_test"):
    def decorator(test_cls):
        this_folder = Path(__file__).parent.absolute()
        test_cls.output_folder = tempfile.TemporaryDirectory()
        output_folder = Path(test_cls.output_folder.name)

        for test_module_name, exec_fn in test_modules:
            src_filepath = this_folder / f"{test_module_name}.py"
            dst_filepath = (output_folder / f"{test_module_name}.py").absolute()
            if not dst_filepath.parent.exists():
                dst_filepath.parent.mkdir(parents=True)
            copy_and_update(src_filepath, dst_filepath)
            test_mod = import_from_path(test_module_name, dst_filepath)
            for attr_name in dir(test_mod):
                if attr_name.startswith("test"):
                    obj = getattr(test_mod, attr_name)
                    if callable(obj):
                        test_name = f"{test_prefix}_{test_module_name.replace('/', '_')}__{attr_name}"
                        def wrapped_test_fn(self, *args, __test_fn__=obj, __exec_fn__=exec_fn, **kwargs):
                            __exec_fn__(__test_fn__)

                        setattr(test_cls, test_name, wrapped_test_fn)
        return test_cls
    return decorator


def multi_threaded(
    num_workers: int,
    num_runs: int = 5,
    skip_tests: Optional[list[str]] = None,
    test_prefix: str = "_original_test",
):
    """Decorator that runs a test in a multi-threaded environment."""
    def decorator(test_cls):
        for name, test_fn in test_cls.__dict__.copy().items():
            if not (name.startswith(test_prefix) and callable(test_fn)):
                continue

            name = f"test{name[len(test_prefix):]}"
            if skip_tests is not None:
                if any(test_name in name for test_name in skip_tests):
                    continue

            def multi_threaded_test_fn(self, capfd, *args, __test_fn__=test_fn, **kwargs):
                barrier = threading.Barrier(num_workers)

                def closure():
                    barrier.wait()
                    for _ in range(num_runs):
                        __test_fn__(self, *args, **kwargs)

                    barrier.wait()
                    gc.collect()
                    assert Context._get_live_count() == 0

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    futures = []
                    for _ in range(num_workers):
                        futures.append(executor.submit(closure))
                    # We should call future.result() to re-raise an exception if test has
                    # failed
                    list(f.result() for f in futures)

                captured = capfd.readouterr()
                if len(captured.err) > 0:
                    if "ThreadSanitizer" in captured.err:
                        raise RuntimeError(f"ThreadSanitizer reported warnings:\n{captured.err}")
                    else:
                        pass
                        # There are tests that write to stderr, we should ignore them
                        # raise RuntimeError(f"Other error:\n{captured.err}")

            setattr(test_cls, f"{name}_multi_threaded", multi_threaded_test_fn)

        return test_cls
    return decorator


@multi_threaded(num_workers=4, num_runs=10)
@add_existing_tests(test_prefix="_original_test")
class TestAllMultiThreaded:
    @pytest.fixture(scope='class')
    def teardown(self):
        self.output_folder.cleanup()

    def _original_test_create_context(self):
        with Context() as ctx:
            print(ctx._get_live_count())
            print(ctx._get_live_module_count())
            print(ctx._get_live_operation_count())
            print(ctx._get_live_operation_objects())
            print(ctx._get_context_again() is ctx)
            print(ctx._clear_live_operations())

    def _original_test_create_module_with_consts(self):
        py_values = [123, 234, 345]
        with Context() as ctx:
            module = Module.create(loc=Location.file("foo.txt", 0, 0))

            dtype = IntegerType.get_signless(64)
            with InsertionPoint(module.body), Location.name("a"):
                arith.constant(dtype, py_values[0])

            with InsertionPoint(module.body), Location.name("b"):
                arith.constant(dtype, py_values[1])

            with InsertionPoint(module.body), Location.name("c"):
                arith.constant(dtype, py_values[2])
