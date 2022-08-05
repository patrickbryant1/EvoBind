#!/bin/bash
set -e
set -x

sudo apt-get update && sudo apt-get install -y build-essential libssl-dev uuid-dev libgpgme11-dev squashfs-tools libseccomp-dev pkg-config git wget
  sudo apt-get install -y build-essential \
  libssl-dev uuid-dev libgpgme11-dev libseccomp-dev \
  pkg-config squashfs-tools cryptsetup


mkdir -p /opt/singularity3/go

export VERSION=1.13.3 OS=linux ARCH=amd64

tmpd=$(mktemp -d)
cd $tmpd

wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz
rm -rf /opt/singularity3
mkdir /opt/singularity3
tar -C /opt/singularity3 -xzf go$VERSION.$OS-$ARCH.tar.gz
export GOPATH=${HOME}/go

export PATH=/opt/singularity3/go/bin:${PATH}:${GOPATH}/bin

mkdir -p $GOPATH/src/github.com/sylabs
cd $GOPATH
rm -rf src/github.com/sylabs
mkdir -p $GOPATH/src/github.com/sylabs
cd $GOPATH/src/github.com/sylabs


git clone https://github.com/sylabs/singularity.git
cd singularity
git checkout v3.5.1

./mconfig --prefix=/opt/singularity3
cd builddir/
make -j 10
make install
singularity --version
