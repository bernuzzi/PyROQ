#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='PyROQ',
    version='Jena',
    description='Builds ROQ data for gravitational waves',
    author='Hong Qi et al.',
    author_email='',
    url = 'https://github.com/bernuzzi/PyROQ',
    packages = find_packages(),
    requires = ['h5py', 'numpy', 'matplotlib']
)
