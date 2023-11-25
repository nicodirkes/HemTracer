# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='hemtracer',
    version='0.1.0',
    description='Lagrangian hemolysis post-processing for Eulerian CFD simulations',
    long_description=readme,
    author='Nico Dirkes',
    author_email='dirkes@aices.rwth-aachen.de',
    # url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'scipy', 'vtk'],
    # packages=['hemtracer', 'hemtracer.hemolysis_model', 'hemtracer.rbc_model'],
    # packages = find_packages(),
)