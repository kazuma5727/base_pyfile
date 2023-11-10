from pathlib import Path

from setuptools import find_packages, setup

import base_pyfile

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory


# mypackage.__version__ を取得する
version = base_pyfile.__version__


def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (str | Path): Path to the requirements.txt file.

    Returns:
        (List[str]): List of parsed requirements.
    """

    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line.split("#")[0].strip()) 

    return requirements


setup(
    name="base_pyfile",
    version=version,
    packages=find_packages(),
    install_requires=parse_requirements(PARENT / "requirements.txt"),
)
