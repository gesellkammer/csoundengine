#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import os

version = (2, 5, 2)

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
    python_requires=">=3.9",
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
        "cachetools",
        "JACK-client",
        "appdirs",
        "pygments",
        "sf2utils",
        "ipywidgets",
        "progressbar2",
        "xxhash",
        "docstring-parser",

        "ctcsound7>=0.4.5",
        "sndfileio>=1.9.0",
        "emlib>=1.14.0",
        "configdict>=2.6.0",
        "bpf4>=1.8.4",
        "numpyx>=1.3.1",
        "pitchtools>=1.10.0",
        "risset>=2.7.7",
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
