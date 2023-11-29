# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from version import _version

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
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
    install_requires=['numpy', 'scipy', 'vtk'],
    python_requires='>=3.12',
)
