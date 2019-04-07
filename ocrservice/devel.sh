#!/bin/bash

start() {
    export FLASK_APP=ocrservice.py
    #export FLASK_DEBUG=1
    python -m flask run --host=0.0.0.0
    #python -m flask run
}

$1 $2 $3 $4 $5