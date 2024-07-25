import datetime
import sys
import platform
import os
import subprocess
from dataclasses import dataclass
from typing import Dict

from .dataclass_utils import config_dataclass_wrapper


def get_current_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "N/A (due to not a git repository)"


def get_version_pip():
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    packages_dict = {}
    lines = result.stdout.split('\n')
    for line in lines[2:]:
        if line:
            package_name, version = line.split()
            packages_dict[package_name] = version
    return packages_dict


def clear_packages(origin_packages: Dict):

    result_packages = {}
    loaded_modules = sys.modules

    for module_name, module in dict(loaded_modules).items():
        if '.' in module_name:
            continue
        if module_name in origin_packages:
            result_packages[module_name] = origin_packages[module_name]
            if hasattr(module, '__version__'):
                result_packages[module_name] = module.__version__

    return result_packages


def get_version(clear_package: bool = False) -> dict:
    packages_dict = get_version_pip()
    if clear_package:
        packages_dict = clear_packages(packages_dict)
    return {
        "_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_git_commit": get_current_commit_hash(),
        "_platform": platform.platform(),
        "_processor": platform.processor(),
        "_python": sys.version,
        "_python_compiler": platform.python_compiler(),
        "_python_packages": packages_dict,
    }


@config_dataclass_wrapper(config_name='version.json')
@dataclass
class VersionInfo:
    _datetime: str
    _git_commit: str
    _platform: str
    _processor: str
    _python: str
    _python_compiler: str
    _python_packages: dict

    @classmethod
    def create(cls, clear_package: bool = False):
        return cls(**get_version(clear_package=clear_package))
