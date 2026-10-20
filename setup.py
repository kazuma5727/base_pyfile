"""Legacy setup.py shim.

パッケージ設定は pyproject.toml に集約している。古いツールや
`python setup.py ...` からもビルドできるようにこのファイルだけ残している。
"""

from setuptools import setup

setup()
