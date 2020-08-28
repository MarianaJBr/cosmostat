from setuptools import find_packages, setup

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Topic :: Scientific/Engineering :: Physics"
]

PACKAGES = find_packages(where="src")
PACKAGE_DIR = {"": "src"}
PACKAGE_DATA = {"": ["*"]}

INSTALL_REQUIRES = [
    "numpy>=1.17",
    "scipy>=1.5.0",
    "matplotlib>=3.3.0",
    # "dataclasses>=0.7;python_version=='3.6.*'",
    "numba>=0.49.0",
    "click>=7.0",
    "dask[bag]>=2.20",
    "h5py>=2.10.0",
    "rich>=5.0.0",
    "toolz>=0.10.0",
]

# Package extras.
EXTRAS_REQUIRE = {
    "test": [
        "pytest>=6.0"
    ],
    "jupyter": [
        "jupyter>=1.0",
        "notebook>=6.0",
        "jupyterlab>=2.0"
    ]
}
EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["test"]

ENTRY_POINTS = {
    "console_scripts": ["chisquarecosmo = chisquarecosmo.cli:main"]
}

SHORT_DESCRIPTION = "Python code to estimate chi-square constraints on " \
                    "cosmology models using background quantities. BAO, " \
                    "SNeIa and CC already implemented. JJE, SEoS, CPL, " \
                    "BA equations of state."

# Get the contents of the README, and use it as the long description.
with open("./README.md") as fp:
    readme_content = fp.read()

setup(name="chisquarecosmo",
      version="0.1.0",
      description=SHORT_DESCRIPTION,
      long_description=readme_content,
      long_description_content_type="text/markdown",
      author="Mariana Jaber, Omar Abel Rodríguez-López",
      author_email="mariana.ifunam@gmail.com, oarodriguez.mx@gmail.com",
      maintainer=None,
      maintainer_email=None,
      url="https://github.com/oarodriguez/chisquarecosmo",
      packages=PACKAGES,
      classifiers=CLASSIFIERS,
      package_dir=PACKAGE_DIR,
      include_package_data=True,
      package_data=PACKAGE_DATA,
      zip_safe=False,
      install_requires=INSTALL_REQUIRES,
      entry_points=ENTRY_POINTS,
      extras_require=EXTRAS_REQUIRE,
      python_requires=">=3.7,<4.0",
      project_urls={
          "Source Code": "https://github.com/oarodriguez/chisquarecosmo"
      })
