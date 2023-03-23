#!/bin/bash

python extract_AM1.py
mv parameters_AM1_MOPAC.csv tmp
sed "s/0.000000/        /g" tmp > parameters_AM1_MOPAC.csv

python extract_PM3.py
mv parameters_PM3_MOPAC.csv tmp
sed "s/0.000000/        /g" tmp > parameters_PM3_MOPAC.csv

python extract_MNDO.py
mv parameters_MNDO_MOPAC.csv tmp
sed "s/0.000000/        /g" tmp > parameters_MNDO_MOPAC.csv
