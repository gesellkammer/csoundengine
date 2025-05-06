from pathlib import Path

def findRoot():
    p = Path(__file__).parent
    if (p/"index.rst").exists():
        return p.parent
    if (p/"pyproject.toml").exists():
        return p
    if (p.parent/"pyproject.toml").exists():
        return p.parent
    raise RuntimeError("Could not locate the root folder")


root = findRoot()
docs = root / "docs"
assert docs.exists()


import csoundengine as ce
cfg = ce.config

rst = cfg.generateRstDocumentation(linkPrefix='config_', withName=False)

outfile = docs / "configkeys.rst"
open(outfile, "w").write(rst)
