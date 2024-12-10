# import os, re, sys

from setuptools import find_packages, setup

setup(
    name = "KittyLM",
    version = "0.0.3.dev0",
    packages = find_packages(include = ["KittyLM", "KittyLM."])
)