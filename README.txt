HOW TO RUN:
    NN:
    Run this command: python3 model/trainModel.py
    
    SVM:
    Run this command: python3 model/SVM.py

    Random Forest:
    Run this command: python3 model/randomForest.py


How Datasets are Preprocessed:
    Our datasets are super huge, so I created a program called dataSubset.py,
    which essentially grabs information from the two most important datasets in the 
    data provided. 

Preprocessing:
    Called Preprocessing.ipyb 
    This file takes the datasets located in the subset folder and processes them.
    This code normalizes the data, and aggregates the data into a single 
    training dataset called 'final.csv'.

    The final.csv file is then used to train the model.

Model:
    The model is located in the model folder.
    model.py contains the model architecture. 
    trainModel.py handles the training, splitting, and k-fold.