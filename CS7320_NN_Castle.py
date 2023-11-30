# Name: James Castle
# Class: CS 7320 Sec 401
# Assignment 7: Neural Networks
# This program represents the creation of a simple neural network that
# predicts success (binary) based on a small set of input data.
import keras
import numpy.random
import keras
from keras import layers
import numpy as np
import pandas as pd
import time
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


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

    seed = 0
    np.random.seed(seed)

    # dataset = df.to_numpy()
    # X = dataset[:, 0:7]
    # Y = dataset[:, 7]

    # attempt to keep things as a pandas DF
    X = df.drop(['success'], axis=1)
    print(X.head())
    Y = df['success']
    print(Y.head())

    # 67% train, 33% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=seed)

    # Create NN and define hyperparams
    model = keras.Sequential()
    model.add(layers.Dense(16, activation='relu'))  # for Part A
    model.add(layers.Dense(16, input_shape=(7,), activation='relu'))  # for Part A
    model.add(layers.Dense(1, activation='sigmoid'))  # output layer
    print('Part A model constructed')
    
    # compile model
    model.compile(loss="binary_crossentropy", optimizer='adam',
                  metrics=['accuracy'])

    tic = time.time()
    # fit model
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # Standardize Data for part B1
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)



    print('-------------START PART B1: STANDARDIZED TEST-------------')
    tic = time.time()
    # fit model
    model.fit(X_train_std, y_train, validation_data=(X_test_std, y_test),
              epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # Check for and remove highly correlated data
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True, cmap='viridis')
    plt.show()

    corrDf = df
    print(corrDf.head())
    corrDf = corrDf.drop(['f1', 'f4'], axis=1)  # drop highly corr
    print(corrDf.head())

    corrMatrix = corrDf.corr()
    sns.heatmap(corrMatrix, annot=True, cmap='viridis')
    plt.show()

    # refit model on de-correlated dataset
    X_corr = corrDf.drop(['success'], axis=1)
    Y_corr = corrDf['success']
    print(X_corr.head())
    print(Y_corr.head())

    # 67% train, 33% test on de-correlated feature dataset
    X_train, X_test, y_train, y_test = train_test_split(X_corr,
                                                        Y_corr,
                                                        test_size=0.33,
                                                        random_state=seed)
    # Standardize Data with correlated features removed for part B2
    scaler = StandardScaler()
    X_train_std_corr = scaler.fit_transform(X_train)
    X_test_std_corr = scaler.transform(X_test)

    model_corr = keras.Sequential()
    model_corr.add(layers.Dense(16, activation='relu'))  # for Part B
    model_corr.add(layers.Dense(16, input_shape=(5,), activation='relu'))  # for Part B
    model_corr.add(layers.Dense(1, activation='sigmoid'))  # output layer

    # compile model
    model_corr.compile(loss="binary_crossentropy", optimizer='adam',
                  metrics=['accuracy'])
    print('Part B2 model constructed')

    print('--------START PART B2: DECORRELATED & STANDARDIZED TEST----')
    tic = time.time()
    # fit model
    model_corr.fit(X_train_std_corr, y_train,
              validation_data=(X_test_std_corr, y_test),
              epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    print('done')

main()


