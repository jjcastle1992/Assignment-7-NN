# Name: James Castle
# Class: CS 7320 Sec 401
# Assignment 7: Neural Networks
# This program represents the creation of a simple neural network that
# predicts success (binary) based on a small set of input data.

import torch
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def main():
    file_name = "7320.finaldata.csv"
    # Read in file
    df = pd.read_csv(file_name, delimiter=",")

    print(df.head())
    print(df.info())

    # data cleaning

    df = df.dropna()  # Drop any rows with missing data
    print(df.info())  # have 3 columns (rank, state not in fl/int format
    df.replace({'rank': {'LO': 0, 'MED': 1, 'HI': 2}}, inplace=True)
    print(df.info())  # verify rank replaced with ints.
    df.replace({'state': {'CAL': 0, 'NY': 1, 'TX': 3}}, inplace=True)
    print(df.info())  # verify state replaced with ints
    print(df.head())


# Create NN and define hyperparams

# run NN

# Print outputs for train-test accuracy

main()


