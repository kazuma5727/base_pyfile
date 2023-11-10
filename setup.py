import re
from pathlib import Path

from setuptools import find_packages, setup

import base_pyfile

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory


# mypackage.__version__ を取得する
version = base_pyfile.__version__
# def get_version():
#     """
#     Retrieve the version number from the 'ultralytics/__init__.py' file.

#     Returns:
#         (str): The version number extracted from the '__version__' attribute in the 'ultralytics/__init__.py' file.
#     """
#     file = PARENT / "base_pyfile/__init__.py"
#     return re.search(
#         r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding="utf-8"), re.M
#     )[1]


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


print(parse_requirements(PARENT / "requirements.txt"))
setup(
    name="base_pyfile",
    version=version,
    packages=find_packages(),
    install_requires=parse_requirements(PARENT / "requirements.txt"),
)
