#!/bin/sh


for document in *.pdf; do
    dst=$(uuidgen -r).txt
    pdftotext "$document" "$dst"
    echo "$document -> $dst" >> generated
done

