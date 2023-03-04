#!/bin/bash

#
# Usage:
# ./upload.sh <user@site> </path/to/remote/doc/folder>
#


set -e
set -x

SSHCONN=$1
DOCPATH=$2

make clean
make html
cd build
scp -r html $SSHCONN:$DOCPATH/pyam-new
CMD="mv $DOCPATH/pyam $DOCPATH/pyam-old && mv $DOCPATH/pyam-new $DOCPATH/pyam && rm -rf $DOCPATH/pyam-old"
ssh $SSHCONN $CMD
