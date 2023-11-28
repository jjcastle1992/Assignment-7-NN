# Name: James Castle
# Class: CS 7320 Sec 401
# Assignment 7: Neural Networks
# This program represents the creation of a simple neural network that
# predicts success (binary) based on a small set of input data.
import keras
import numpy.random
import tensorflow as tf
import keras
from keras import layers
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
    df.replace({'state': {'CAL': 0, 'NY': 1, 'TX': 2}}, inplace=True)
    print(df.info())  # verify state replaced with ints
    print(df.head(10))

    dataset = df.to_numpy()
    seed = 0
    numpy.random.seed(seed)

    X = dataset[:, 0:7]
    Y = dataset[:, 7]
    print('done')

    # 67% train, 33% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=seed)

    # Create NN and define hyperparams
    model = keras.Sequential()
    model.add(layers.Dense(16, input_shape=(7,), activation='relu'))  # for Part A
    model.add(layers.Dense(1, activation='sigmoid'))  # output layer
    print('Part A model constructed')
    
    # compile model
    model.compile(loss="binary_crossentropy", optimizer='adam',
                  metrics=['accuracy'])

    # fit model
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=32)
    # run NN

    # Print outputs for train-test accuracy

main()


