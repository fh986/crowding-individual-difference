"""
Setup script for the Correlation Attenuation Toolbox package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="attenuation_toolbox",
    version="1.0.0",
    author="Correlation Attenuation Toolbox Contributors",
    author_email="",
    description="A toolkit for correcting correlation coefficients for attenuation due to measurement unreliability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/attenuation-toolbox",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scipy>=1.4.0",
        "matplotlib>=3.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "jupyter>=1.0",
        ],
    },
)
