#!/bin/bash
set -x
set -e

# get conda and install it
if [[ "$TRAVIS_OS_NAME" != 'windows' ]]; then
    echo "Starting download from $URL"
    wget $URL -O miniconda.sh --no-check-certificate
    echo "Download complete from $URL"
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda
else
    choco install $CHOCONAME --params='"/AddToPath"'
    choco install make
fi

# update conda
conda update --yes conda

# create named env
conda create -n testing python=$PYVERSION --yes

# install deps, specific versions are used to guarantee consistency with
# plotting tests
conda install -n testing --yes \
      numpy \
      pandas \
      matplotlib \
      seaborn \
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
      gdal \
      fiona \
      geopandas \
      cartopy
