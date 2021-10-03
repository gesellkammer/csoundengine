#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

version = (1, 5, 2)

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
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "JACK-client",
        "ctcsound",
        "sndfileio>=1.7",
        "emlib>=1.1",
        "cachetools",
        "configdict>=0.8",
        "bpf4",
        "numpyx>=0.5",
        "pitchtools>=1.1",
        "appdirs",
        "pygments",
        "sf2utils",
        "jupyter",
        "ipywidgets"
    ],
    license="BSD",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
    package_data={'': ['data/*']},
    include_package_data=True,
    zip_safe=False
)
