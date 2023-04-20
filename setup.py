from setuptools import find_packages, setup

import base_pyfile

# mypackage.__version__ を取得する
version = base_pyfile.__version__

requirements = []
with open("requirements.txt", "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        requirements.append(line)

setup(
    name="base_pyfile",
    version=version,
    packages=find_packages(),
    install_requires=requirements,
)
