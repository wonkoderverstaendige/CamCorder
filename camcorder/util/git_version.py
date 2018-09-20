import os
import contextlib
from pathlib import Path
import subprocess as sp


def git_version():
    with working_directory(str(Path(__file__).parent)):
        try:
            version = sp.check_output(["git", "describe", "--always"]).strip().decode('utf-8')
        except sp.CalledProcessError:
            version = "Unknown"
    return version


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = str(Path.cwd())
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
