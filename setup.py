"""
Custom setup.py that compiles the C++ game library during wheel creation.

Uses the DummyExtension pattern (from original OpenAI procgen) to trigger
build_ext, which produces a platform-tagged wheel and copies the compiled
shared library into the wheel's package data.
"""

import importlib.util
import os
import platform
import shutil
import sys
import types

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(SCRIPT_DIR, "procgen_gym")

LIB_NAMES = {
    "Windows": "env.dll",
    "Darwin": "libenv.dylib",
    "Linux": "libenv.so",
}


class DummyExtension(Extension):
    """Empty extension that forces setuptools to produce a platform wheel."""

    def __init__(self):
        super().__init__("dummy", sources=[])


def _load_module(name, filepath):
    """Load a module directly from a file path without package imports."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class BuildCppExt(build_ext):
    """Custom build_ext that compiles the C++ library via CMake."""

    def run(self):
        if self.inplace:
            print("skipping inplace build, library will be built on demand")
            return

        # Register a stub procgen_gym package with a minimal libenv shim
        # so builder.py's `from .libenv import get_header_dir` resolves
        # without pulling in numpy/ctypes (unavailable in build env).
        stub = types.ModuleType("procgen_gym")
        stub.__path__ = [PACKAGE_DIR]
        sys.modules["procgen_gym"] = stub

        libenv_shim = types.ModuleType("procgen_gym.libenv")
        libenv_shim.get_header_dir = lambda: os.path.join(PACKAGE_DIR, "src")
        sys.modules["procgen_gym.libenv"] = libenv_shim

        builder_mod = _load_module(
            "procgen_gym.builder", os.path.join(PACKAGE_DIR, "builder.py")
        )

        lib_dir = builder_mod.build(package=True)
        lib_name = LIB_NAMES[platform.system()]
        src = os.path.join(lib_dir, lib_name)

        if not os.path.exists(src):
            raise RuntimeError(f"Build succeeded but library not found at {src}")

        # Copy into the build staging area (not the source tree)
        dst_dir = os.path.join(self.build_lib, "procgen_gym", "data", "prebuilt")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, lib_name))
        print(f"Copied {lib_name} to {dst_dir}")


setup(
    ext_modules=[DummyExtension()],
    cmdclass={"build_ext": BuildCppExt},
)
