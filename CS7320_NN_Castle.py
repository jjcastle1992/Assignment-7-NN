# Name: James Castle
# Class: CS 7320 Sec 401
# Assignment 7: Neural Networks
# This program represents the creation of a simple neural network that
# predicts success (binary) based on a small set of input data.

import keras
from keras import layers
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
    df.replace({'rank': {'LO': 0, 'MED': 1, 'HI': 2}},
               inplace=True)
    print(df.info())  # verify rank replaced with ints.
    df.replace({'state': {'CAL': 0, 'NY': 1, 'TX': 2}},
               inplace=True)
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
    model.add(layers.Dense(16, input_shape=(7,),
                           activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print('Part A model constructed')
    
    # compile model
    model.compile(loss="binary_crossentropy", optimizer='adam',
                  metrics=['accuracy'])

    print('--------START PART A: Single Layer no std or decorr DF----')
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


    # *************** PART B **********

    print('-------------START PART B1: STANDARDIZED TEST-------------')
    tic = time.time()
    # fit model
    model.fit(X_train_std, y_train,
              validation_data=(X_test_std, y_test), epochs=100,
              batch_size=32)
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
    model_corr.add(layers.Dense(16, input_shape=(5,),
                                activation='relu'))  # for Part B
    model_corr.add(layers.Dense(1, activation='sigmoid'))

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

    # *************** PART C ***************************************

    # *************** EXP C1 ***************************************
    # define a new model
    modelC1 = keras.Sequential()
    modelC1.add(layers.Dense(16, input_shape=(7,),
                            activation='relu'))
    modelC1.add(layers.Dense(16, activation='relu'))
    modelC1.add(layers.Dense(1, activation='sigmoid'))

    modelC1.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C1 model constructed')

    print('----START PART C1: STANDARDIZED TEST w/ 2 hidden layers----')
    # fit new model with 7 inputs
    tic = time.time()
    modelC1.fit(X_train_std, y_train,
                validation_data=(X_test_std, y_test),
                epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C2 ***************************************
    # Define Model C2 on 5 inputs (decorrelated) w/ 2 hidden layers
    modelC2 = keras.Sequential()
    modelC2.add(layers.Dense(16, input_shape=(5,),
                             activation='relu'))
    modelC2.add(layers.Dense(16, activation='relu'))
    modelC2.add(layers.Dense(1, activation='sigmoid'))

    modelC2.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C2 model constructed')

    print('---START PART C2: DECORRELATED & STD w/ 2 hidden layers----')

    # fit new model with 5 feature inputs (decorrelated)
    tic = time.time()
    modelC2.fit(X_train_std_corr, y_train,
                validation_data=(X_test_std_corr, y_test),
                epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C3 ***************************************
    # C3 Revisit correlation (Drop only F4)
    modelC3 = keras.Sequential()
    modelC3.add(layers.Dense(16, input_shape=(6,),
                             activation='relu'))
    modelC3.add(layers.Dense(16, activation='relu'))
    modelC3.add(layers.Dense(1, activation='sigmoid'))

    modelC3.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C3 model constructed')

    noF4df = df
    noF4df = noF4df.drop(['f4'], axis=1)
    print(noF4df.head())

    # fit model on de-correlated dataset (F4 Dropped only)
    X_corr_noF4 = noF4df.drop(['success'], axis=1)
    Y_corr_noF4 = noF4df['success']
    print(X_corr_noF4.head())
    print(Y_corr_noF4.head())

    # 67% train, 33% test on de-correlated feature dataset
    X_train, X_test, y_train, y_test = train_test_split(X_corr_noF4,
                                                        Y_corr_noF4,
                                                        test_size=0.33,
                                                        random_state=seed)
    # Standardize Data with correlated features removed for part C3
    scaler = StandardScaler()
    X_train_nof4 = scaler.fit_transform(X_train)
    X_test_nof4 = scaler.transform(X_test)

    print('---START PART C3: No F4, Std,  w/ 2 hidden layers----')
    tic = time.time()
    modelC3.fit(X_train_nof4, y_train,
                validation_data=(X_test_nof4, y_test),
                epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C4 ***************************************
    # C4 Revisit correlation (Drop only F1)

    modelC4 = keras.Sequential()
    modelC4.add(layers.Dense(16, input_shape=(6,),
                             activation='relu'))
    modelC4.add(layers.Dense(16, activation='relu'))
    modelC4.add(layers.Dense(1, activation='sigmoid'))

    modelC4.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C4 model constructed')

    noF1df = df
    noF1df = noF1df.drop(['f1'], axis=1)
    print(noF1df.head())

    # fit model on de-correlated dataset (F1 Dropped only)
    X_corr_noF1 = noF1df.drop(['success'], axis=1)
    Y_corr_noF1 = noF1df['success']
    print(X_corr_noF1.head())
    print(Y_corr_noF1.head())

    # 67% train, 33% test on de-correlated feature dataset
    X_train, X_test, y_train, y_test = train_test_split(X_corr_noF1,
                                                        Y_corr_noF1,
                                                        test_size=0.33,
                                                        random_state=seed)
    # Standardize Data with correlated features removed for part C4
    scaler = StandardScaler()
    X_train_nof1 = scaler.fit_transform(X_train)
    X_test_nof1 = scaler.transform(X_test)

    print('---START PART C4: No F1, Std,  w/ 2 hidden layers----')
    tic = time.time()
    modelC4.fit(X_train_nof1, y_train,
                validation_data=(X_test_nof1, y_test),
                epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C5 ***************************************
    # C5: No dropping of correlated data. Double Epochs
    modelC5 = keras.Sequential()
    modelC5.add(layers.Dense(16, input_shape=(7,),
                            activation='relu'))
    modelC5.add(layers.Dense(16, activation='relu'))
    modelC5.add(layers.Dense(1, activation='sigmoid'))

    modelC5.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C5 model constructed')

    print('----START PART C5: 2 hidden layers. 200 epochs----')
    # fit new model with 7 inputs
    tic = time.time()
    modelC5.fit(X_train_std, y_train,
                validation_data=(X_test_std, y_test),
                epochs=200, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C6 ***************************************
    # C6: No dropping of correlated data. Halve Epochs
    modelC6 = keras.Sequential()
    modelC6.add(layers.Dense(16, input_shape=(7,),
                            activation='relu'))
    modelC6.add(layers.Dense(16, activation='relu'))
    modelC6.add(layers.Dense(1, activation='sigmoid'))

    modelC6.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C6 model constructed')

    print('----START PART C6: 2 hidden layers. 50 epochs----')
    # fit new model with 7 inputs
    tic = time.time()
    modelC6.fit(X_train_std, y_train,
                validation_data=(X_test_std, y_test),
                epochs=50, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C7 ***************************************
    # C7: Double batch size to 64. No correlation drops. 100 Epochs
    modelC7 = keras.Sequential()
    modelC7.add(layers.Dense(16, input_shape=(7,),
                             activation='relu'))
    modelC7.add(layers.Dense(16, activation='relu'))
    modelC7.add(layers.Dense(1, activation='sigmoid'))

    modelC7.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C7 model constructed')

    # 67% train, 33% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C7: 64 batch size. 100 epochs----')
    # fit new model with 7 inputs
    tic = time.time()
    modelC7.fit(X_train_std, y_train,
                validation_data=(X_test_std, y_test),
                epochs=100, batch_size=64)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C8 ***************************************
    # C8: batch size to 32. No correlation drops. 100 Epochs.
    # 50: 50 Test: Train
    modelC8 = keras.Sequential()
    modelC8.add(layers.Dense(16, input_shape=(7,),
                             activation='relu'))
    modelC8.add(layers.Dense(16, activation='relu'))
    modelC8.add(layers.Dense(1, activation='sigmoid'))

    modelC8.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C8 model constructed')

    # 50% train, 50% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.5,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C8: 50:50 test-train----')
    tic = time.time()
    modelC8.fit(X_train_std, y_train,
                validation_data=(X_test_std, y_test),
                epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C9 ***************************************
    # C9: batch size to 32. No correlation drops. 100 Epochs.
    # 20:80 Test: Train
    modelC9 = keras.Sequential()
    modelC9.add(layers.Dense(16, input_shape=(7,),
                             activation='relu'))
    modelC9.add(layers.Dense(16, activation='relu'))
    modelC9.add(layers.Dense(1, activation='sigmoid'))

    modelC9.compile(loss="binary_crossentropy", optimizer='adam',
                    metrics=['accuracy'])
    print('Part C9 model constructed')

    # 20% train, 80% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.8,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C9: 20:80 test-train----')
    tic = time.time()
    modelC9.fit(X_train_std, y_train,
                validation_data=(X_test_std, y_test),
                epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C10 ***************************************
    # C10: batch size to 32. No correlation drops. 100 Epochs.
    # 80:20 Test: Train
    modelC10 = keras.Sequential()
    modelC10.add(layers.Dense(16, input_shape=(7,),
                             activation='relu'))
    modelC10.add(layers.Dense(16, activation='relu'))
    modelC10.add(layers.Dense(1, activation='sigmoid'))

    modelC10.compile(loss="binary_crossentropy", optimizer='adam',
                     metrics=['accuracy'])
    print('Part C10 model constructed')

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C10: 80:20 test-train----')
    tic = time.time()
    modelC10.fit(X_train_std, y_train,
                 validation_data=(X_test_std, y_test),
                 epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C11 ***************************************
    # C11: batch size to 32. No correlation drops. 100 Epochs.
    # 32 second layer neurons
    modelC11 = keras.Sequential()
    modelC11.add(layers.Dense(16, input_shape=(7,),
                              activation='relu'))
    modelC11.add(layers.Dense(32, activation='relu'))
    modelC11.add(layers.Dense(1, activation='sigmoid'))

    modelC11.compile(loss="binary_crossentropy", optimizer='adam',
                     metrics=['accuracy'])
    print('Part C11 model constructed')

    # 67% train, 33% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C11: 32 second layer neurons test----')
    tic = time.time()
    modelC11.fit(X_train_std, y_train,
                 validation_data=(X_test_std, y_test),
                 epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C12 ***************************************
    # C12: batch size to 32. No correlation drops. 100 Epochs.
    # C1 but a Third layer of 16 neurons.
    modelC12 = keras.Sequential()
    modelC12.add(layers.Dense(16, input_shape=(7,),
                              activation='relu'))
    modelC12.add(layers.Dense(16, activation='relu'))
    modelC12.add(layers.Dense(16, activation='relu'))
    modelC12.add(layers.Dense(1, activation='sigmoid'))

    modelC12.compile(loss="binary_crossentropy", optimizer='adam',
                     metrics=['accuracy'])
    print('Part C12 model constructed')

    # 67% train, 33% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C12: 3rd layer of 16 neurons test----')
    tic = time.time()
    modelC12.fit(X_train_std, y_train,
                 validation_data=(X_test_std, y_test),
                 epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C13 ***************************************
    # C13: batch size to 32. No correlation drops. 100 Epochs.
    # C1 but use SGD for the optimizer.
    modelC13 = keras.Sequential()
    modelC13.add(layers.Dense(16, input_shape=(7,),
                              activation='relu'))
    modelC13.add(layers.Dense(16, activation='relu'))
    modelC13.add(layers.Dense(1, activation='sigmoid'))

    modelC13.compile(loss="binary_crossentropy", optimizer='sgd',
                     metrics=['accuracy'])
    print('Part C13 model constructed')

    # 67% train, 33% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C13: SGD Optimizer test----')
    tic = time.time()
    modelC13.fit(X_train_std, y_train,
                 validation_data=(X_test_std, y_test),
                 epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C14 ***************************************
    # C14: batch size to 32. No correlation drops. 100 Epochs.
    # C1 but use set first hidden layer to 8 neurons
    modelC14 = keras.Sequential()
    modelC14.add(layers.Dense(8, input_shape=(7,),
                              activation='relu'))
    modelC14.add(layers.Dense(16, activation='relu'))
    modelC14.add(layers.Dense(1, activation='sigmoid'))

    modelC14.compile(loss="binary_crossentropy", optimizer='adam',
                     metrics=['accuracy'])
    print('Part C14 model constructed')

    # 67% train, 33% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C14: 8 hidden layer neurons test----')
    tic = time.time()
    modelC14.fit(X_train_std, y_train,
                 validation_data=(X_test_std, y_test),
                 epochs=100, batch_size=32)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    # *************** EXP C15 ***************************************
    # C15: batch size to 16. No correlation drops. 80 Epochs.
    # Set Train:test to 30:70.
    # Set optimizer to Nadam

    modelC15 = keras.Sequential()
    modelC15.add(layers.Dense(16, input_shape=(7,),
                              activation='relu'))
    modelC15.add(layers.Dense(16, activation='relu'))
    modelC15.add(layers.Dense(1, activation='sigmoid'))

    modelC15.compile(loss="binary_crossentropy", optimizer='Nadam',
                     metrics=['accuracy'])
    print('Part C15 model constructed')

    # 30% train, 70% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.70,
                                                        random_state=seed)

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print('----START PART C15: custom test----')
    tic = time.time()
    modelC15.fit(X_train_std, y_train,
                 validation_data=(X_test_std, y_test),
                 epochs=80, batch_size=16)
    toc = time.time()
    print(f'{(toc - tic):.04f} seconds elapsed.')

    print('done')

    """
    Epoch 75/80
    2/2 [==============================] - 0s 27ms/step - loss: 0.2619 - accuracy: 0.9655 - val_loss: 0.3936 - val_accuracy: 0.8551
    Epoch 76/80
    2/2 [==============================] - 0s 27ms/step - loss: 0.2566 - accuracy: 0.9655 - val_loss: 0.3918 - val_accuracy: 0.8551
    Epoch 77/80
    2/2 [==============================] - 0s 26ms/step - loss: 0.2514 - accuracy: 0.9655 - val_loss: 0.3902 - val_accuracy: 0.8551
    Epoch 78/80
    2/2 [==============================] - 0s 28ms/step - loss: 0.2457 - accuracy: 0.9655 - val_loss: 0.3888 - val_accuracy: 0.8551
    Epoch 79/80
    2/2 [==============================] - 0s 28ms/step - loss: 0.2406 - accuracy: 0.9655 - val_loss: 0.3875 - val_accuracy: 0.8406
    Epoch 80/80
    2/2 [==============================] - 0s 28ms/step - loss: 0.2359 - accuracy: 0.9655 - val_loss: 0.3863 - val_accuracy: 0.8406
    3.1469 seconds elapsed.
    """

main()


