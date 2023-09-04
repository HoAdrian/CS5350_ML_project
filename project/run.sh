#!/bin/bash

cd perceptron
python3 perceptron.py
cd ..
cd ensemble
python3 bagging.py
python3 adaboost.py
cd ..
cd svm
python3 svm/svm.py
cd ..
cd logistic_regression
python3 logistic_regression/logistic_regression.py
cd ..