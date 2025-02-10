"""Usual setup file for package"""

# read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
install_requires = (this_directory / "requirements.txt").read_text().splitlines()

with open("gradboost_pv/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, version = line.replace("'", "").split()
            version = version.replace('"', "")


setup(
    name="gradboost_pv",
    version=version,
    license="MIT",
    description="Repository for inference and training of XGBoost based National PV Model.",
    author="Tom Armstrong, Peter Dudfield, Tom Armstrong",
    author_email="info@openclimatefix.org",
    company="Open Climate Fix Ltd",
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(),
)
