#!/bin/bash
set -x
set -e

# get conda and install it
if [[ "$TRAVIS_OS_NAME" != 'windows' ]]; then
    URL="https://repo.anaconda.com/miniconda/Miniconda$PYVERSION-latest-$OSNAME-x86_64.$EXT"
    echo "Starting download from $URL"
    wget $URL -O miniconda.sh --no-check-certificate
    echo "Download complete from $URL"
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda
else
    choco install $CHOCONAME --params="'/AddToPath:1'"
    echo "foo"
    ls /c/tools
    echo "bar"
    ls /c/ProgramData/chocolatey/lib/Miniconda
    echo "baz"
    grep -i 'miniconda' /c/ProgramData/chocolatey/logs/chocolatey.log
    echo $PATH
    which conda
fi

# update conda
conda update --yes conda

# create named env
conda create -n testing python=$PYVERSION --yes

# install deps, specific versions are used to guarantee consistency with
# plotting tests
conda install -n testing --yes \
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
conda install -n testing -c conda-forge --yes \
      libkml \
      gdal \
      fiona \
      geopandas \
      cartopy
