from setuptools import setup

# Version
version = None
with open("soothsayer_utils/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, "Check version in soothsayer_utils/__init__.py"

setup(
name='soothsayer_utils',
    version=version,
    description='Utility functions for Soothsayer',
    url='https://github.com/jolespin/soothsayer_utils',
    author='Josh L. Espinoza',
    author_email='jespinoz@jcvi.org',
    license='BSD-3',
    packages=["soothsayer_utils"],
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "pathlib2",
        "tqdm >=4.19",
        "tzlocal",
        "bz2file",
        ## Optional: 
        # "biopython",
        # "mmh3",
        # "matplotlib",
      ],
)
