import datetime
import sys
import platform
import os
import subprocess


def get_current_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "N/A (due to not a git repository)"


def get_version():
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    packages_dict = {}
    lines = result.stdout.split('\n')
    for line in lines[2:]:
        if line:
            package_name, version = line.split()
            packages_dict[package_name] = version
    return {
        "_datetime": datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S"),
        "_git_commit": get_current_commit_hash(),
        "_platform": platform.platform(),
        "_processor": platform.processor(),
        "_python": sys.version,
        "_python_compiler": platform.python_compiler(),
        "_python_packages": packages_dict,
    }
