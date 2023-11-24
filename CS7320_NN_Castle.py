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


def read_file(file_name):
    try:
        with open(file_name, 'r') as file:
            data = list(csv.reader(file, delimiter=','))
            file.close()
            data = np.array(data)
            return data

    except FileNotFoundError:
        return []

def data_cleaning(data):
    # delete lines with null values
    pass
    # replace non-numrical data with numerical values

def main():
    file_name = "7320.finaldata.csv"
    # Read in file
    raw_data = read_file(file_name)
    print(raw_data.shape)



# Data Wrangling

# Create NN and define hyperparams

# run NN

# Print outputs for train-test accuracy

main()


