#!/bin/bash

if [[ -z "${VIRTUAL_ENV}" ]]
then
    PYTHON_PATH="python"
else
    PYTHON_PATH="${VIRTUAL_ENV}/bin/python"
fi

SCRIPT_DIR=${0%/*}

for dir in "${SCRIPT_DIR}"/ssepoptim/extensions/*/setup.py
do
    cd "${dir%/*}"
    "$PYTHON_PATH" setup.py install
done
