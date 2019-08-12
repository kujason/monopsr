#!/usr/bin/env bash

source "${HOME}/.virtualenvs/monopsr/bin/activate"

cd "$(dirname "$0")"
cd ../..

#echo "Running example unit tests"
#coverage run --source monopsr -m unittest discover -b --pattern "test_*.py"
#unittest discover -b --pattern "test_*.py"

echo "Running unit tests in $(pwd)"
coverage run --source monopsr -m unittest discover -b --pattern "*_test.py"

#coverage report -m
