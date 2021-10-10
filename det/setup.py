"""Package setup."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bdd100k-models-det",  # Replace with your own username
    version="1.0.0",
    author="Fisher Yu",
    author_email="i@yf.io",
    description="BDD100K Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.bdd100k.com/",
    project_urls={
        "Documentation": "https://doc.bdd100k.com/",
        "Source": "https://github.com/SysCV/bdd100k-models",
        "Tracker": "https://github.com/bdd100k/bdd100k/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "bdd100k @ git+git://github.com/bdd100k/bdd100k.git",
        "mmcv-full",
        "mmdet",
    ],
)
