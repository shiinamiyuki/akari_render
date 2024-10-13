import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import subprocess

CPP_EXT_DIR = Path("cpp_extension/")
CPP_EXT_DEPS_DIR = CPP_EXT_DIR / "ext"

def clone_dep(url: str, revision: str, dest: Path):
    if dest.exists():
        print(f"Skipping cloning {url} to {dest} as it already exists")
        return
    subprocess.run(["git", "clone", '--no-checkout', url, dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", revision], check=True)
    subprocess.run(["git", "-C", dest, "submodule", "update",
                   "--init", "--recursive"], check=True)

def build_cpp_ext():
    clone_dep("https://github.com/RenderKit/embree", "v4.3.3",CPP_EXT_DEPS_DIR / "embree")


def build_impl(profile: str):
    pass


def build():
    pass


def main():
    build_cpp_ext()
    parser = ArgumentParser(
        prog="run.py", description="AkariRender build/run utility")


if __name__ == '__main__':
    main()
