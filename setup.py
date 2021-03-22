#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

readme = open('README.rst').read()
version = (0, 1, 0)

setup(
    name='csoundengine',
    python_requires=">=3.8",
    version=".".join(map(str, version)),
    description='A synthesis framework using csound',
    long_description=readme,
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
        "pitchtools>=1.0"
    ],
    license="BSD",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: LGPL2 License',
    ],
)
