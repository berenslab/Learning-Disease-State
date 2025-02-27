#!/usr/bin/python3

import setuptools

with open("README.md", "r", encoding='utf8') as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="src",
    version="0.0.1",
    description="Learning-Disease-Severity-from-Noisy-Ordinal-Disease-Progression-Labels",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
)