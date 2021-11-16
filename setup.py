#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
import os
import io


def read_file(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding='utf8').read()


setup(name="rolewalk",
      version="0.0.1",
      description=("Structural node embedding on directed graphs"
                   " using bidirectional random walks"),
      long_description=read_file('README.md'),
      py_modules=["rolewalk"],
      author="Maixent Chenebaux",
      author_email="max.chbx@gmail.com",
      maintainer="Maixent Chenebaux",
      maintainer_email="max.chbx@gmail.com",
      url="https://github.com/kerighan/rolewalk",
      download_url="http://pypi.python.org/pypi/rolewalk",
      keywords="graph, node, embedding, structural, random walks",
      platforms=["Linux", "Mac OSX", "Windows", "Unix"],
      classifiers=[
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Scientific/Engineering :: Bio-Informatics",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
      ])
