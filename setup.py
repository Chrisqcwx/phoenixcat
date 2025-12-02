from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="phoenixcat",
    version="0.5.3",
    author="Hongyao Yu, Sijin Yu",
    author_email="chrisqcwx@gmail.com, eeyusijin@mail.scut.edu.cn",
    description="An enhanced infrastructure library for PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chrisqcwx/phoenixcat",
    package_dir={"": "src"},
    packages=find_packages("src"),
    license="Apache 2.0 License",
    python_requires=">=3.8",
    install_requires=["torch", "torchvision", "diffusers"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
    ],
)
