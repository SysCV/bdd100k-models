#! /bin/bash

FOLDER=$1

python3 -m black $FOLDER
python3 -m isort $FOLDER
python3 -m pylint $FOLDER
python3 -m flake8 --docstring-convention google $FOLDER
python3 -m mypy --install-types --strict --allow-untyped-calls --show-error-codes $FOLDER
