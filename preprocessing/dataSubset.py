import os 
import pandas as pd
import numpy as np


class dataSetSet():
    def __init__(self):
        self.filePath = os.path.dirname(__file__)
        self.dataFolder = os.path.join(self.filePath, '../data/')
        self.saveFolder = os.path.join(self.dataFolder, 'subset')
        self.creditFolder = "home-credit-credit-risk-model-stability"

    """
    Opens the dataframe, and takes a random n number of rows 
    and saves it to a new file
    filename is train_base.csv, train_applprev_1_0.csv, ...
    SAMPLE: dataSubset(n=100, filename='train_base.csv', random_state=42)
    """
    def dataSubset(self, n, filename, random_state=42):
        creditData = os.path.join(self.dataFolder, self.creditFolder, 'csv_files', 'train')
        df = pd.read_csv(os.path.join(creditData, filename))
        df = df.sample(n=n, random_state=random_state)

        # Now save the data into the subset folder
        if not os.path.exists(self.saveFolder):
            os.makedirs(self.saveFolder)
        df.to_csv(os.path.join(self.saveFolder, filename), index=False)

    """
    Opens two files in subset folder, and merges the data frames 
    on the primary key of 'case_id' 
    """
    def joinDataSubsets(self, **csvFiles):
        # Join the data subsets into one file
        print("TODO")

