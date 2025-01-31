import os
import sys
import subprocess
from typing import List


def run_process(command: List[str], cwd: str):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    print(stderr.decode())
    if process.returncode != 0:
        raise RuntimeError(
            f'Error running {command}, exit code {process.returncode:0xd}')


def run_maturin(profile: str, verbose: str | None):
    print("Running maturin...")
    cmds = ['maturin', 'develop', '--color', 'always', f'--profile={profile}']
    if verbose:
        cmds.append(verbose)
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = cur_file_dir + '/crates/pyakari/'
    run_process(cmds, cwd)


if __name__ == '__main__':
    profile = 'dev'
    verbose = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--release':
            profile = 'release'
        if arg == '--dev':
            profile = 'dev'
        if arg == 'profile':
            profile = sys.argv[i+2]
        if arg == '--verbose' or arg.startswith('-v'):
            verbose = arg
    print(f'Building with profile {profile}')
    run_maturin(profile, verbose)
