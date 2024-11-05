#!/bin/bash

source kaggle_v2_env/bin/activate
pip install ipykernel
python -m ipykernel install --user --name kaggle_v2_env --display-name "Python 3.10 (kaggle_v2)"