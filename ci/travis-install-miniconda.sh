#!/bin/bash
set -x
set -e

# get directory of this script, thanks https://stackoverflow.com/a/246128
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

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

