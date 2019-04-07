#!/bin/bash

test() {
    PROJ_PATH=..
    cd $PROJ_PATH
    curl -F image=@'data/table_small.jpg' http://localhost:5000/ocr
}

$1 $2 $3 $4 $5
