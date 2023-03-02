#!/usr/bin/env bash

DIR=${1:-$HOME/.cache/models}
if [ ! -f $DIR/wiki.en.bin ]; then
    mkdir -p $DIR
    [ -f $DIR/wiki.en.zip ] || wget -O $DIR/wiki.en.zip "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip"
    unzip $DIR/wiki.en.zip -d $DIR
fi
