import os
from setuptools import setup, find_packages
from src.utilsbox import add_welcome_msg

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

add_welcome_msg(os.getlogin(), 'Gook luck!')

setup(
    name='utilsbox',
    version="0.0.0",
    author="Hongyao Yu, Sijin Yu, Zijiao Chen",
    description="Utils for python and pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chrisqcwx/utilsbox-python",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[],
)
