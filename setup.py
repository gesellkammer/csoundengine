#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import os

version = (1, 18, 0)

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


datafiles = package_files('csoundengine/data')


setup(
    name='csoundengine',
    python_requires=">=3.10",
    version=".".join(map(str, version)),
    description='A synthesis framework using csound',
    long_description=long_description,
    author='Eduardo Moguillansky',
    author_email='eduardo.moguillansky@gmail.com',
    url='https://github.com/gesellkammer/csoundengine',
    packages=[
        'csoundengine',
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "JACK-client",
        "ctcsound",
        "sndfileio>=1.8.2",
        "emlib>=1.7.7",
        "cachetools",
        "configdict>=2.4",
        "bpf4",
        "numpyx>=0.5",
        "pitchtools>=1.9.2",
        "appdirs",
        "pygments",
        "sf2utils",
        "jupyter",
        "ipywidgets",
        "risset",
        "progressbar2",
        "xxhash",
        "docstring-parser"
    ],
    license="BSD",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
    package_data={'csoundengine': datafiles},
    include_package_data=True,
    zip_safe=False
)
