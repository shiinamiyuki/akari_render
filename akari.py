from dataclasses import dataclass
import os
import stat
import sys
from argparse import ArgumentParser
from pathlib import Path
import subprocess
from typing import List
import shutil
from enum import Enum, auto

CPP_EXT_DIR = Path("cpp_extension/")
CPP_EXT_DEPS_DIR = CPP_EXT_DIR / "ext"
ALL_DEPS: List['Dependency'] = []
CMAKE_BUILD_DIR = Path("cmake-build")
EMBREE_BUILD_DIR = Path("cmake-build-embree")
DLL_EXT = ".so" if sys.platform == "linux" else ".dll"
DLL_PREFIX = "lib" if sys.platform == "linux" else ""


class Dependency:
    name: str
    url: str
    revision: str
    dest: Path

    def __init__(self, name: str, url: str, revision: str, dest: Path):
        self.name = name
        self.url = url
        self.revision = revision
        self.dest = dest

    def download(self):
        if self.dest.exists():
            return
        cprint(
            f"Downloading {self.url} to {self.dest}", Color.CYAN)
        subprocess.run(["git", "clone", '--no-checkout',
                       self.url, self.dest], check=True)
        subprocess.run(["git", "-C", self.dest, "checkout",
                       self.revision], check=True)
        subprocess.run(["git", "-C", self.dest, "submodule", "update",
                        "--init", "--recursive"], check=True)


ALL_DEPS.append(Dependency("embree", "https://github.com/RenderKit/embree",
                           "v4.3.3", CPP_EXT_DEPS_DIR / "embree"))
ALL_DEPS.append(Dependency("vcpkg", "https://github.com/microsoft/vcpkg",
                           "2024.09.30", CPP_EXT_DIR / "vcpkg"))


class Color(Enum):
    RED = auto()
    GREEN = auto()
    YELLOW = auto()
    BLUE = auto()
    MAGENTA = auto()
    CYAN = auto()
    WHITE = auto()


def cprint(s: str, color: Color = Color.WHITE):
    match color:
        case Color.RED:
            print(f"\033[91m{s}\033[0m")
        case Color.GREEN:
            print(f"\033[92m{s}\033[0m")
        case Color.YELLOW:
            print(f"\033[93m{s}\033[0m")
        case Color.BLUE:
            print(f"\033[94m{s}\033[0m")
        case Color.MAGENTA:
            print(f"\033[95m{s}\033[0m")
        case Color.CYAN:
            print(f"\033[96m{s}\033[0m")
        case Color.WHITE:
            print(f"\033[97m{s}\033[0m")


def download_deps() -> None:
    for dep in ALL_DEPS:
        dep.download()


def cache_args() -> List[str]:
    args: List[str] = []
    # check if sccache or ccache is available
    if shutil.which("sccache") is not None:
        args = ["-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache"]
    elif shutil.which("ccache") is not None:
        args = ["-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
    return args


def build_embree() -> None:
    """
    Building embree separately
    This is due to compiling embree with MSVC is too slow
    """
    cprint('Building Embree', color=Color.CYAN)
    if not (EMBREE_BUILD_DIR / "CMakeCache.txt").exists():
        subprocess.run(["cmake", "-G", "Ninja", "-B",
                        "cmake-build-embree", "cpp_extension/ext/embree",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                        "-DCMAKE_C_COMPILER=clang",
                        "-DCMAKE_CXX_COMPILER=clang++",
                        "-DEMBREE_TASKING_SYSTEM=INTERNAL",
                        "-DEMBREE_ISPC_SUPPORT=OFF",
                        "-DEMBREE_GEOMETRY_QUAD=OFF",
                        "-DEMBREE_GEOMETRY_SUBDIVISION=OFF",
                        "-DEMBREE_TUTORIALS=OFF",
                        "-DCMAKE_INSTALL_PREFIX=cmake-build-embree/install",
                        "-DEMBREE_STATIC_LIB=ON",
                        ] + cache_args(), check=True)
    subprocess.run(["cmake", "--build", "cmake-build-embree"], check=True)
    subprocess.run(["cmake", "--install", "cmake-build-embree"], check=True)


def configure_cmake() -> None:
    cprint('Configuring CMake', color=Color.CYAN)
    compiler_path_args: List[str] = []
    if os.name == 'nt':
        # find MSVC
        ret = subprocess.run(
            ["cargo", "run", "--bin", "find_msvc"], check=True,  stdout=subprocess.PIPE)
        # get MSVC path from output
        msvc_path = ret.stdout.decode().strip()
        if msvc_path == 'MSVC not found':
            print("MSVC not found")
            sys.exit(1)
        cl_link_path = msvc_path.split("\n")
        cl_path = cl_link_path[0]
        link_path = cl_link_path[1]
        compiler_path_args = ["-DCMAKE_C_COMPILER=" + cl_path,
                              "-DCMAKE_CXX_COMPILER=" + cl_path,
                                "-DCMAKE_LINKER=" + link_path]
    subprocess.run(["cmake", "-G", "Ninja", "-B",
                    "cmake-build", "cpp_extension",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
                    ] + compiler_path_args + cache_args(), check=True)


def build_cpp_ext() -> None:
    cprint("Downloading dependencies", color=Color.CYAN)
    download_deps()
    build_embree()
    cprint('Running CMake', color=Color.CYAN)
    if not (CMAKE_BUILD_DIR / "CMakeCache.txt").exists():
        configure_cmake()
    cprint('Building C++ extension', color=Color.CYAN)
    subprocess.run(["cmake", "--build", "cmake-build"], check=True)


def build_impl(profile: str):
    build_cpp_ext()
    subprocess.run(["cargo", "build", "--profile", profile], check=True)


class CargoArgs:
    profile: str
    bin_args: List[str]

    def __init__(self, profile: str, bin_args: List[str]):
        self.profile = profile
        self.bin_args = bin_args

    @ staticmethod
    def parse(args: List[str]):
        bin_args_start = args.index("--") if "--" in args else None
        if bin_args_start is not None:
            bin_args = args[bin_args_start + 1:]
            args = args[:bin_args_start]
        else:
            bin_args = []
        profile = "dev"
        parser = ArgumentParser()
        parser.add_argument("--profile", default="dev")
        parser.add_argument("--release", action="store_true")
        parsed = parser.parse_args(args)
        if parsed.release:
            profile = "release"
        else:
            profile = parsed.profile
        return CargoArgs(profile, bin_args)


def build(args: List[str]):
    cargo_args = CargoArgs.parse(args)
    build_impl(cargo_args.profile)


def run(args: List[str]):
    if len(args) == 0:
        binary = "akari_cli"
    else:
        binary = args[0]
        args = args[1:]
    cargo_args = CargoArgs.parse(args)
    build_impl(cargo_args.profile)
    # set up environment variables
    env = os.environ.copy()
    env["AKARI_CPP_EXT_LIB"] = str(
        CMAKE_BUILD_DIR / f'{DLL_PREFIX}akari_cpp_ext{DLL_EXT}')
    cprint(f"Running {binary}", color=Color.CYAN)
    subprocess.run(["cargo", "run", "-q", "--profile",
                   cargo_args.profile, "--bin", binary], check=False, env=env)


def rmtree(path: Path):
    def onerror(func, path, exc_info):
        if not os.path.exists(path):
            return
        os.chmod(path, stat.S_IWUSR)
        func(path)
    shutil.rmtree(path, onerror=onerror)


def rmfile(path: Path):
    if not os.path.exists(path):
        return
    os.chmod(path, stat.S_IWUSR)
    os.remove(path)


def clean(args: List[str]):
    # remove all build artifacts
    mode = 'all'
    if len(args) > 1:
        print("Invalid number of arguments")
        sys.exit(1)
    if len(args) == 1:
        mode = args[0]
    match mode:
        case 'all':
            shutil.rmtree(CMAKE_BUILD_DIR, ignore_errors=True)
            subprocess.run(["cargo", "clean"], check=True)
        case 'cache':
            rmfile(CMAKE_BUILD_DIR / "CMakeCache.txt")
        case 'cmake':
            shutil.rmtree(CMAKE_BUILD_DIR, ignore_errors=True)
        case 'cargo':
            subprocess.run(["cargo", "clean"], check=True)
        case 'deps':
            # ask user for confirmation
            print("This will remove following directories:")
            for dep in ALL_DEPS:
                print(f'{dep.name} ({dep.dest})')
            print("Do you want to continue? [y/N]")
            response = input()
            if response.upper() == 'Y':
                for dep in ALL_DEPS:
                    rmtree(dep.dest)
            else:
                print("Aborting")
                sys.exit(1)


def main():
    # get python version
    if sys.version_info < (3, 11):
        print("Python 3.11 or later is required")
        sys.exit(1)

    args = sys.argv[1:]

    def print_help():
        print("AkariRender utility script")
        cprint("  Usage: akari.py <command> ...")
        print("Commands:")
        print("  cmake")
        print("  build [--profile <profile>] [--release] [-q]")
        print("  run <binary> [--profile <profile>] [--release] [-q] [-- ...]")
        print("  clean [all|cache|cmake|cargo]")
    if len(args) == 0:
        print_help()
        sys.exit(1)
    cmd = args[0]
    match cmd:
        case 'cmake':
            configure_cmake()
        case 'build':
            build(args[1:])
        case 'run':
            run(args[1:])
        case 'clean':
            clean(args[1:])
        case _:
            print_help()


if __name__ == '__main__':
    main()
