import os
from setuptools import setup

with open("README.md", "r") as fp:
    long_description = fp.read()

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
setup(
    name='daceml',
    version='0.1.0a',
    url='https://github.com/spcl/dace-onnx',
    author='SPCL @ ETH Zurich',
    author_email='rauscho@ethz.ch',
    description='DaCe frontend for machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=['daceml'],
    package_data={'': ['*.cpp']},
    install_requires=[
        'dace@git+https://github.com/spcl/dace.git@0e3f2ec', 'onnx == 1.7.0',
        'torch', 'dataclasses; python_version < "3.7"'
    ],
    # install with pip and --find-links (see Makefile)
    # See https://github.com/pypa/pip/issues/5898
    extras_require={
        'testing': [
            'coverage', 'pytest', 'yapf', 'pytest-cov', 'transformers',
            'pytest-xdist', 'torchvision'
        ],
        'docs': [
            'sphinx==3.2.1', 'sphinx_rtd_theme==0.5.2',
            'sphinx-autodoc-typehints==1.11.1', 'sphinx-gallery==0.9.0',
            'matplotlib==3.4.2'
        ],
        'debug': ['onnxruntime']
    })
