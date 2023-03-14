from setuptools import find_packages, setup

setup(
    name="base_pyfile",
    version="0.0.4",
    packages=find_packages(),
    install_requires=["natsort", "tqdm"],
)
