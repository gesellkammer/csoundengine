#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

version = (0, 2, 0)

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='csoundengine',
    python_requires=">=3.8",
    version=".".join(map(str, version)),
    description='A synthesis framework using csound',
    long_description=long_description,
    author='Eduardo Moguillansky',
    author_email='eduardo.moguillansky@gmail.com',
    url='https://github.com/gesellkammer/csoundengine',
    packages=[
        'csoundengine',
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "JACK-client",
        "ctcsound",
        "sndfileio",
        "emlib",
        "cachetools",
        "configdict>=0.4",
        "bpf4",
        "numpyx>=0.5",
        "pitchtools>=1.0",
        "sounddevice"
    ],
    license="BSD",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
)
