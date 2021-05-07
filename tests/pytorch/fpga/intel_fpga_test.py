#!/usr/bin/env python3
# This module has been inspired by the testing infrastructure in DaCe: https://github.com/spcl/dace
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import click
import os
from pathlib import Path
import re
import subprocess as sp
import sys
from typing import Any, Iterable, Union

TEST_TIMEOUT = 600  # Seconds

from fpga_testing import (Colors, DACE_DIR, TEST_DIR, cli, dump_logs,
                          print_status, print_success, print_error)

# (relative path, sdfg name(s), run synthesis, args to executable)
# Whenever is supported, the "-test" flag enable more extensive tests
TESTS = [
    ("pytorch/fpga/test_gemm_fpga.py", "dace_model", ["-test"]),
    ("pytorch/fpga/test_im2col_conv2d_fpga.py", "dace_model", ["-test"]),
    ("pytorch/fpga/test_matmul_fpga.py", "dace_model", ["-test"]),
    ("pytorch/fpga/test_maxpool2d_fpga.py", "dace_model", []),
    ("pytorch/fpga/test_relu_fpga.py", "dace_model", []),
    ("pytorch/fpga/test_reshape_fpga.py", "dace_model", ["-test"]),
    ("pytorch/fpga/test_softmax_fpga.py", "dace_model", []),

    # Multi Head Attention
    ("pytorch/fpga/test_attn_fpga.py", "dace_model", []),

    # Streaming composition test
    ("pytorch/fpga/test_streaming_conv_relu_mp.py", "dace_model", []),
]


def run(path: Path, sdfg_names: Union[str, Iterable[str]],
        args: Iterable[Any]):

    # Set environment variables
    os.environ["DACE_compiler_fpga_vendor"] = "intel_fpga"
    os.environ["DACE_compiler_use_cache"] = "0"
    os.environ["DACE_compiler_default_data_types"] = "C"
    # We would like to use DACE_cache=hash, but we want to have access to the
    # program's build folder
    # TODO: enable when DaCeML-Dace version is updated
    # os.environ["DACE_cache"] = "name"
    os.environ["DACE_compiler_intel_fpga_mode"] = "emulator"
    os.environ["DACE_optimizer_transform_on_call"] = "0"
    os.environ["DACE_optimizer_interface"] = ""
    os.environ["DACE_optimizer_autooptimize"] = "0"

    path = DACE_DIR / path
    if not path.exists():
        print_error(f"Path {path} does not exist.")
        return False
    base_name = f"{Colors.UNDERLINE}{path.stem}{Colors.END}"

    if isinstance(sdfg_names, str):
        sdfg_names = [sdfg_names]
    for sdfg_name in sdfg_names:
        build_folder = TEST_DIR / ".dacecache" / sdfg_name / "build"
        if build_folder.exists():
            # There is a potential conflict between the synthesis folder
            # generated by Xilinx and the one generated by Intel FPGA
            sp.run(["make", "clean"],
                   cwd=build_folder,
                   stdout=sp.PIPE,
                   stderr=sp.PIPE,
                   check=True,
                   timeout=60)

    # Simulation in software
    print_status(f"{base_name}: Running emulation.")

    try:
        proc = sp.Popen(map(str, [sys.executable, path] + args),
                        cwd=TEST_DIR,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE,
                        encoding="utf-8")
        sim_out, sim_err = proc.communicate(timeout=TEST_TIMEOUT)
    except sp.TimeoutExpired:
        dump_logs(proc)
        print_error(f"{base_name}: Emulation timed out "
                    f"after {TEST_TIMEOUT} seconds.")
        return False
    if proc.returncode != 0:
        dump_logs((sim_out, sim_err))
        print_error(f"{base_name}: Emulation failed.")
        return False
    print_success(f"{base_name}: Emulation successful.")

    for sdfg_name in sdfg_names:
        build_folder = TEST_DIR / ".dacecache" / sdfg_name / "build"
        if not build_folder.exists():
            print_error(f"Invalid SDFG name {sdfg_name} for {base_name}.")
            return False
        open(build_folder / "simulation.out", "w").write(sim_out)
        open(build_folder / "simulation.err", "w").write(sim_err)

    return True


@click.command()
@click.option("--parallel/--no-parallel", default=True)
@click.argument("tests", nargs=-1)
def intel_fpga_cli(parallel, tests):
    """
    If no arguments are specified, runs all tests. If any arguments are
    specified, runs only the tests specified (matching on file name or SDFG
    name).
    """
    cli(TESTS, run, tests, parallel)


if __name__ == "__main__":
    intel_fpga_cli()