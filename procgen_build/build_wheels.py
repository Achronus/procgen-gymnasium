"""
Build platform wheels locally and upload to a GitHub Release.

CI then picks up the wheels from the release and publishes to PyPI.

Usage:
    python -m procgen_build                              # build Windows + Linux
    python -m procgen_build --platform win               # Windows only
    python -m procgen_build --platform linux             # Linux only (requires Docker)
    python -m procgen_build --upload v0.1.1              # upload wheels to GitHub Release
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WHEELHOUSE = os.path.join(PROJECT_ROOT, "wheelhouse")


def _find_qt_bin_dir():
    """Locate the directory containing Qt5Core.dll on Windows."""
    # 1. vcpkg in project root
    vcpkg_bin = os.path.join(PROJECT_ROOT, "vcpkg_installed", "x64-windows", "bin")
    if os.path.isfile(os.path.join(vcpkg_bin, "Qt5Core.dll")):
        return vcpkg_bin

    # 2. PROCGEN_CMAKE_PREFIX_PATH env var — walk up to find bin/
    cmake_prefix = os.environ.get("PROCGEN_CMAKE_PREFIX_PATH", "")
    if cmake_prefix:
        candidate = cmake_prefix
        for _ in range(4):
            bin_dir = os.path.join(candidate, "bin")
            if os.path.isfile(os.path.join(bin_dir, "Qt5Core.dll")):
                return bin_dir
            candidate = os.path.dirname(candidate)

    # 3. Search PATH
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and os.path.isfile(os.path.join(d, "Qt5Core.dll")):
            return d

    return None


def _find_qt_cmake_dir():
    """Locate the Qt5 CMake config directory on Windows."""
    # vcpkg
    vcpkg_cmake = os.path.join(
        PROJECT_ROOT, "vcpkg_installed", "x64-windows", "share", "cmake"
    )
    if os.path.isdir(vcpkg_cmake):
        return vcpkg_cmake

    # From env var
    cmake_prefix = os.environ.get("PROCGEN_CMAKE_PREFIX_PATH", "")
    if cmake_prefix and os.path.isdir(cmake_prefix):
        return cmake_prefix

    return None


def _clean_prebuilt():
    """Remove stale libraries from data/prebuilt/ so they don't leak into other platform wheels."""
    prebuilt = os.path.join(PROJECT_ROOT, "procgen_gym", "data", "prebuilt")
    if os.path.isdir(prebuilt):
        shutil.rmtree(prebuilt)


def _run(cmd, env=None):
    """Run a command, printing it first."""
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def build_windows():
    """Build a Windows wheel using cibuildwheel (native)."""
    print("=" * 60)
    print("Building Windows wheel")
    print("=" * 60)
    _clean_prebuilt()

    qt_bin = _find_qt_bin_dir()
    qt_cmake = _find_qt_cmake_dir()

    if not qt_bin:
        print("ERROR: Could not find Qt5Core.dll. Install Qt5 via vcpkg or set PROCGEN_CMAKE_PREFIX_PATH.")
        sys.exit(1)

    print(f"Qt bin dir: {qt_bin}")
    print(f"Qt cmake dir: {qt_cmake}")

    env = os.environ.copy()
    env["CIBW_BUILD"] = "cp313-win_amd64"
    env["CIBW_BEFORE_BUILD"] = "pip install delvewheel"
    env["CIBW_REPAIR_WHEEL_COMMAND_WINDOWS"] = (
        f'delvewheel repair -w {{dest_dir}} {{wheel}}'
        f' --add-path "{qt_bin}" --analyze-existing'
    )
    if qt_cmake:
        env["PROCGEN_CMAKE_PREFIX_PATH"] = qt_cmake

    _run(["cibuildwheel", "--output-dir", WHEELHOUSE, "--platform", "windows"], env=env)


def build_linux():
    """Build a Linux wheel using cibuildwheel (Docker)."""
    print("=" * 60)
    print("Building Linux wheel (via Docker)")
    print("=" * 60)
    _clean_prebuilt()

    # Check Docker is available
    if not shutil.which("docker"):
        print("ERROR: Docker is required for Linux wheel builds. Install Docker Desktop and ensure it's running.")
        sys.exit(1)

    env = os.environ.copy()
    env["CIBW_BUILD"] = "cp313-manylinux_x86_64"
    env["CIBW_SKIP"] = "*-musllinux*"
    env["CIBW_BEFORE_BUILD_LINUX"] = (
        "yum install -y qt5-qtbase-devel cmake || "
        "(apt-get update && apt-get install -y qtbase5-dev cmake)"
    )
    env["CIBW_ENVIRONMENT"] = "PROCGEN_CMAKE_PREFIX_PATH=/usr"

    _run(["cibuildwheel", "--output-dir", WHEELHOUSE, "--platform", "linux"], env=env)


def upload_to_release(tag):
    """Upload wheels to a GitHub Release so CI can publish them to PyPI."""
    print("=" * 60)
    print(f"Uploading wheels to GitHub Release {tag}")
    print("=" * 60)

    if not shutil.which("gh"):
        print("ERROR: GitHub CLI (gh) is required. Install from https://cli.github.com/")
        sys.exit(1)

    wheels = [f for f in os.listdir(WHEELHOUSE) if f.endswith(".whl")]
    if not wheels:
        print("ERROR: No wheels found in wheelhouse/. Build first.")
        sys.exit(1)

    print(f"Found {len(wheels)} wheel(s):")
    for w in wheels:
        print(f"  {w}")

    wheel_paths = [os.path.join(WHEELHOUSE, w) for w in wheels]
    _run(["gh", "release", "upload", tag, "--clobber", *wheel_paths])


def main():
    parser = argparse.ArgumentParser(description="Build procgen-gym wheels locally")
    parser.add_argument(
        "--platform",
        choices=["win", "linux", "all"],
        default="all",
        help="Which platform to build for (default: all)",
    )
    parser.add_argument(
        "--upload",
        metavar="TAG",
        help="Upload wheels to a GitHub Release (e.g. --upload v0.1.1)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove wheelhouse/ before building",
    )
    args = parser.parse_args()

    # Ensure cibuildwheel is installed
    if not shutil.which("cibuildwheel"):
        print("Installing cibuildwheel...")
        _run([sys.executable, "-m", "pip", "install", "cibuildwheel"])

    if args.clean and os.path.exists(WHEELHOUSE):
        shutil.rmtree(WHEELHOUSE)
    os.makedirs(WHEELHOUSE, exist_ok=True)

    if args.platform in ("win", "all"):
        if platform.system() != "Windows" and args.platform == "win":
            print("ERROR: Windows wheels can only be built on Windows.")
            sys.exit(1)
        if platform.system() == "Windows":
            build_windows()

    if args.platform in ("linux", "all"):
        build_linux()

    if args.upload:
        upload_to_release(args.upload)

    print("\nDone! Wheels in:", WHEELHOUSE)


if __name__ == "__main__":
    main()
