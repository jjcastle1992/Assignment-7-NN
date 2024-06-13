# Assignment 7: Neural Networks
Name: James Castle

Course: CS7320

Date of last update: Dec 2, 2023

Purpose:
This program represents the creation of a simple neural network that
predicts success (binary) based on a small set of input data.

How to Use:
1. Ensure you have brought over the following files into a single directory:
   * CS7320_NN_Castle.ipynb (this is your primary interface, the .py is an outdated file)
   * 7320.finaldata.csv (the small dataset the models will use to train and test on)
   * Ensure you have the following libraries installed:
     * keras
     * matplotlib
     * numpy
     * pandas
     * matplotlib
     * seaborn
     * sklearn
2. Run the CS7320_NN_Castle.ipynb either sequentially from top to bottom
   or feel free to run the whole workbook at once.


The main components of the CS7320_NN_Castle Jupyter Notebook are:
  * Read in of data from the 7320.finaldata.csv file
  * Data cleaning (dropping records with missing info, converting fields to ints)
  * Setup Training and validation sets
  * Creation and Running/Timing of:
    * Model A (no standardization or removal of highly correlated data)
    * Model B1 (standardized data without removal of highly correlated data)
    * Model B2 (Standardized Data with removal of correlated data)
    * Part C1: using standardized but non-decorrelated data. Same as B1, but with a second 16 neuron hidden layer 
    * Part C2: using standardized and decorrelated data. Same as B2, but with a second 16 neuron hidden layer 
    * Part C3: using standardized decorrelated data where only F4 was removed (Same as C2, but only removed F4)
    * Part C4: using standardized decorrelated data where only F1 was removed (Same as C2, but only removed F1)
    * Part C5: use standardized non-decorrelated data. Double Epochs 
    * Part C6: use standardized non-decorrelated data. Halve Epochs 
    * Part C7: double batch size from 32 to 64 
    * Part C8: batch size back to 32 and set train-test split to 50:50 instead of 67:33. 
    * Part C9: set train-test split to 20:80 instead of 67:33. 
    * Part C10: set train-test split to 80:20 instead of 67:33. 
    * Part C11: C1, but with double the 2nd layer neurons 
    * Part C12: C1, but with a 3rd layer of 16 neurons 
    * Part C13: C1, but with SGD as the optimizer 
    * Part C14: C1 but Change first hidden layer to 8 neurons. 
    * Part C15: Change optimizer to Nadam. Set train:test to 30:70. Epochs to 80. Batch size 16. 
    * Part C16: General tinkering with various hyperparameters and model setup to get the best results.
