# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from version import _version
from pathlib import Path

this_directory = Path(__file__).parent
with open(this_directory / 'README.md', encoding='utf-8') as f:
    readme = f.read()

with open(this_directory / 'LICENSE', encoding='utf-8') as f:
    license = f.read()

setup(
    name='hemtracer',
    version=_version,
    description='Lagrangian hemolysis post-processing for Eulerian CFD simulations',
    long_description=readme,
    author='Nico Dirkes',
    author_email='dirkes@aices.rwth-aachen.de',
    url='nicodirkes.github.io/HemTracer/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'scipy', 'vtk', 'pandas'],
    python_requires='>=3.12',
)
