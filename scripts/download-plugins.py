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
pluginsroot = Path("/home/em/dev/python/csoundengine/data/plugins")
tmpfile, _ = urllib.request.urlretrieve(infourl)
info = json.load(open(tmpfile))
assets = info.get('assets')
asseturls = [asset['browser_download_url'] for asset in assets]
destfolder = "/home/em/Downloads"
for url in asseturls:
    zipped = download(url, destfolder)
    unzipped = extract(zipped)
    if "linux" in url:
        subfolder, globpatt = "linux", "*.so"
    elif "macos" in url:
        subfolder, globpatt = "macos", "*.dylib"
    elif "windows" in url or "win64" in url:
        subfolder, globpatt = "windows", "*.dll"
    else:
        print("Url unknown", url)    
        continue
    plugins = list(unzipped.glob(globpatt))
    copyfiles(plugins, pluginsroot/subfolder)
        
      