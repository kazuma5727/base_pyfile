from setuptools import find_packages, setup

import base_pyfile

# mypackage.__version__ を取得する
version = base_pyfile.__version__
requirements = base_pyfile.read_text_file("requirements.txt", "\n")

setup(
    name="base_pyfile",
    version=version,
    packages=find_packages(),
    install_requires=requirements,
)
