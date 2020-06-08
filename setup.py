from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["tensorflow>=1"]

setup(
    name="tracepooling-layer",
    version="0.0.1",
    author="Irene Martin",
    author_email="irene.martin@uv.es",
    description="A package to implement trace pooling layer",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/spatuvlab/tracepooling-layer/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)