from setuptools import find_packages, setup

setup(
    name="base_pyfile",
    version="0.0.2",
    packages=find_packages(),
    install_requires=["natsort", "tqdm"],
    # entry_points={"console_scripts": ["my_command=my_package.my_module:main"]},
)
