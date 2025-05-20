from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

# Define the C extension with optimization flags
c_extension = Extension(
    'near.process_near_results',
    sources=['src/c/process_near_results.c'],
    include_dirs=['src/c'],
    extra_compile_args=['-O3'],  # Add -O3 optimization
)

setup(
    name="near",
    version="0.1.0",
    packages=['near'],
    ext_modules=[c_extension],
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'tqdm',
        'pyyaml',
        'faiss-gpu-cuvs==1.11.0'
    ],
    entry_points={
        'console_scripts': [
            'near=near.main:main',
        ],
    },
    python_requires='>=3.11.11',
)