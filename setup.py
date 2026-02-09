"""
Custom setup.py that compiles the C++ game library during wheel creation.

This ensures platform-specific wheels include the prebuilt shared library
(env.dll / libenv.so / libenv.dylib) so users don't need a C++ toolchain.
"""

import importlib.util
import os
import platform
import shutil
import sys
import types

from setuptools import setup, Distribution
from setuptools.command.build_py import build_py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(SCRIPT_DIR, "procgen_gym")
PREBUILT_DIR = os.path.join(PACKAGE_DIR, "data", "prebuilt")

LIB_NAMES = {
    "Windows": "env.dll",
    "Darwin": "libenv.dylib",
    "Linux": "libenv.so",
}


def _load_module(name, filepath):
    """Load a module directly from a file path without package imports."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class BinaryDistribution(Distribution):
    """Force setuptools to produce a platform-specific wheel."""

    def has_ext_modules(self):
        return True


class BuildWithCpp(build_py):
    """Custom build_py that compiles the C++ library first."""

    def run(self):
        self._build_cpp()
        super().run()

    def _build_cpp(self):
        # Register a stub procgen_gym package so relative imports in
        # builder.py and libenv.py resolve in pip's isolated build env.
        stub = types.ModuleType("procgen_gym")
        stub.__path__ = [PACKAGE_DIR]
        sys.modules["procgen_gym"] = stub

        _load_module("procgen_gym.libenv", os.path.join(PACKAGE_DIR, "libenv.py"))
        builder_mod = _load_module(
            "procgen_gym.builder", os.path.join(PACKAGE_DIR, "builder.py")
        )

        lib_dir = builder_mod.build(package=True)
        lib_name = LIB_NAMES[platform.system()]
        src = os.path.join(lib_dir, lib_name)

        if not os.path.exists(src):
            raise RuntimeError(f"Build succeeded but library not found at {src}")

        os.makedirs(PREBUILT_DIR, exist_ok=True)
        shutil.copy2(src, os.path.join(PREBUILT_DIR, lib_name))
        print(f"Copied {lib_name} to {PREBUILT_DIR}")


setup(
    cmdclass={"build_py": BuildWithCpp},
    distclass=BinaryDistribution,
)
