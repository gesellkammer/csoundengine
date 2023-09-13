#!/usr/bin/env python3
"""
Downloads latest release of the needed csound plugins from github
and extracts them to the csoundengine packege so that they can be
distributed

"""
from __future__ import annotations
import urllib.request
import json
import os
import zipfile
import tempfile
import shutil
from pathlib import *


def download(url, destfolder) -> Path:
    fileName = os.path.split(url)[1]
    dest = Path(destfolder) / fileName
    if dest.exists():
        print(f"Destination {dest} already exists, overwriting")
        os.remove(dest)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"   ... saved to {dest}")
    return dest


def extract(zipped:Path) -> Path:
    destFolder = tempfile.mktemp(prefix=zipped.name)
    os.mkdir(destFolder)
    with zipfile.ZipFile(zipped, 'r') as z:
        z.extractall(destFolder)
    return Path(destFolder)
    

def copyfiles(files, dest:Path):
    if not dest.exists():
        os.makedirs(dest.as_posix(), exist_ok=True)
    for f in files:
        print(f"Copying {f} to {dest}")
        shutil.copy(f, dest)


infourl = "https://api.github.com/repos/csound-plugins/csound-plugins/releases/latest"
dataroot = Path("/home/em/dev/python/csoundengine/csoundengine/data")
tmpfile, _ = urllib.request.urlretrieve(infourl)
info = json.load(open(tmpfile))
assets = info.get('assets')
asseturls = [asset['browser_download_url'] for asset in assets]
destfolder = "/home/em/Downloads"
for url in asseturls:
    zipped = download(url, destfolder)
    unzipped = extract(zipped)
    if 'csound7' in url:
        pluginsfolder = 'plugins7'
    else:
        pluginsfolder = 'plugins6'

    if "linux.zip" in url:
        subfolder, globpatt = "linux-x86_64", "*.so"
    elif "macos.zip" in url:
        subfolder, globpatt = "macos-x86_64", "*.dylib"
    elif "macos-arm64" in url:
        subfolder, globpatt = "macos-arm64", "*.dylib"
    elif "windows.zip" in url or "win64" in url:
        subfolder, globpatt = "windows-x86_64", "*.dll"
    else:
        print("Url unknown", url)
        continue

    plugins = list(unzipped.glob(globpatt))
    copyfiles(plugins, dataroot/pluginsfolder/subfolder)
