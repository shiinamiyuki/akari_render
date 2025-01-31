import os
import sys
import subprocess
from typing import List
import threading

def run_process(command: List[str], cwd: str):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    # stdout, stderr = process.communicate()
    def print_output(pipe):
        for line in iter(pipe.readline, b''):
            print(line.decode(), end='')

    stdout_thread = threading.Thread(target=print_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr,))
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f'Error running {command}, exit code {process.returncode:0x}')


def run_maturin(profile: str, verbose: str | None):
    print("Running maturin...")
    cmds = ['maturin', 'develop', '--color', 'always', f'--profile={profile}']
    if verbose:
        cmds.append(verbose)
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = cur_file_dir + '/crates/pyakari/'
    print(' '.join(cmds))
    run_process(cmds, cwd)

def fix_config_toml():
    python_interpreter_path = sys.executable.replace('\\', '/')
    pyo3_python_line = f'PYO3_PYTHON="{python_interpreter_path}" # Automatically added by build.py'
    if not os.path.exists('.cargo'):
        os.makedirs('.cargo')
    if not os.path.exists('.cargo/config.toml'):
        with open('.cargo/config.toml', 'w') as f:
            f.write('')
    with open('.cargo/config.toml', 'r') as f:
        config = f.read()
        if '[env]' not in config:
            config += '\n[env]\n'
        lines = config.split('\n')
        env_line_idx = None
        pyo3_line_idx = None
        for i, line in enumerate(lines):
            if line.startswith('PYO3_PYTHON'):
                pyo3_line_idx = i
            if line == '[env]':
                env_line_idx = i
        assert env_line_idx is not None
        if pyo3_line_idx is None:
            lines.insert(env_line_idx + 1, pyo3_python_line)
        else:
            lines[pyo3_line_idx] = pyo3_python_line
        new_config = '\n'.join(lines)
        if new_config != config:
            print('Updating .cargo/config.toml')
            print('writing:', pyo3_python_line)
            with open('.cargo/config.toml', 'w') as f:
                f.write(new_config)
        
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
    fix_config_toml()
    print(f'Building with profile {profile}')
    run_maturin(profile, verbose)
