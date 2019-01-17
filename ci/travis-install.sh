#!/bin/bash
set -x
set -e

# setup miniconda url
case "${TRAVIS_OS_NAME}" in
    linux)
        OSNAME=Linux
        EXT=sh
    ;;
    osx)
        OSNAME=MacOSX
        EXT=sh
    ;;
    windows)
        OSNAME=Windows
        EXT=exe
    ;;
esac

case "${PYENV}" in
    py27)
        PYVERSION=2
    ;;
    py37)
        PYVERSION=3
    ;;
esac

# get conda
URL="https://repo.anaconda.com/miniconda/Miniconda$PYVERSION-latest-$OSNAME-x86_64.$EXT"
echo "Starting download from $URL"
wget $URL -O miniconda.sh;
echo "Download complete from $URL"

# install and update conda
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda

# create named env
conda create -n pyam-testing python=$PYVERSION --yes

# install deps, specific versions are used to guarantee consistency with
# plotting tests
conda install -n pyam-testing --yes \
      numpy==1.14.0 \
      pandas==0.22.0 \
      matplotlib==2.1.2 \
      seaborn==0.8.1 \
      six \
      pyyaml \
      xlrd \
      xlsxwriter \
      requests \
      jupyter \
      nbconvert \
      pandoc \
      pytest \
      coveralls \
      pytest-cov 

# these have to be installed from conda-forge to get right gdal packages
conda install -n pyam-testing -c conda-forge --yes \
      libkml \
      fiona \
      geopandas==0.3.0 \
      cartopy==0.16.0
