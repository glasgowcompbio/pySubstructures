
#!/usr/bin/env python
import os
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "ms2lda", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="pySubstructures",
    version=version["__version__"],
    description=" Python package to perform unsupervised discoveries of motifs from tandem mass spectrometry data.",
    long_description_content_type="text/markdown",
    long_description=readme,
    author="Joe Wandy, Niek de Jonge",
    author_email="",
    url="https://github.com/glasgowcompbio/pySubstructures",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    test_suite="tests",
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "seaborn",
        "scikit-learn",
        "matplotlib",
        "plotly",
        "loguru",
        "jupyterlab",
        "ipywidgets",
        "tqdm",
        "networkx",
        "gensim",
        "pip",
        "tqdm",
        "pymzml==2.4.7",
        "mass-spec-utils",
        "pyMolNetEnhancer"],
    extras_require={}
)