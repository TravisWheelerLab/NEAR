from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os

# Define the C extension with optimization flags
c_extension = Extension(
    'near.process_near_results',
    sources=['src/c/main.c',
             'src/c/types.c',
             'src/c/io.c',
             'src/c/process_hits.c',
             'src/c/util.c'],
    include_dirs=['src/c'],
    extra_compile_args=['-O3'],
)

setup(
    name="near",
    version="0.1.0",
    package_dir={"": "src"},  # Add this line
    packages=find_packages(where="src"),  # Modify this line
    ext_modules=[c_extension],
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'tqdm',
        'pyyaml',
    ],
    entry_points={
        'console_scripts': [
            'near=near.main:main',
        ],
    },
    python_requires='>=3.11.11',
)
