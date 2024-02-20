import os 
import pandas as pd
import numpy as np


class dataSetSet():
    def __init__(self):
        self.filePath = os.path.dirname(__file__)
        self.dataFolder = os.path.join(self.filePath, '..', 'data')
        self.saveFolder = os.path.join(self.dataFolder, 'subset')
        self.creditFolder = "home-credit-credit-risk-model-stability"

    def getTrainData(self):
        return os.path.join(self.dataFolder, self.creditFolder, 'csv_files', 'train')

    """
    Opens the dataframe, and takes a random number of rows 
    and saves it to a new file in the subset folder.
    filename is train_base.csv, train_applprev_1_0.csv, ...
    SAMPLE: dataSubset(n=100, filename='train_base.csv', random_state=42)
    """
    def dataSubset(self, n, filename, random_state=42):
        df = pd.read_csv(os.path.join(self.getTrainData(), filename))
        df = df.sample(n=n, random_state=random_state)
        if not os.path.exists(self.saveFolder):
            os.makedirs(self.saveFolder)
        df.to_csv(os.path.join(self.saveFolder, filename), index=False)

    """
    Opens two files in subset folder, and merges the data frames 
    on the primary key of 'case_id' 
    """
    def joinDataSubsets(self, **csvFiles):
        # Join the data subsets into one file
        # First, get the train_base.csv file
        train_basePath = os.path.join(self.saveFolder, 'train_base.csv')
        if not os.path.exists(train_basePath):
            print('train_base.csv does not exist in subset folder')
            print('Creating one automatically...')
            self.dataSubset(n=100, filename='train_base.csv', random_state=42)

        # Now for each file, merge it with the train_base.csv file
        train_base_df = pd.read_csv(train_basePath)
        for key, value in csvFiles.items():
            if not os.path.exists(os.path.join(self.saveFolder, value)):
                print(f'{value} does not exist in subset folder')
                print('Creating one automatically...')
                self.dataSubset(n=100, filename=value, random_state=42)
            df = pd.read_csv(os.path.join(self.saveFolder, value))
            train_base_df = train_base_df.merge(df, on='case_id')


dataTrainNames = {
    'aplprev0': 'train_applprev_1_0.csv', 'aplprev1': 'train_applprev_1_1.csv',
    'aplprev2': 'train_applprev_2.csv',
    'bureau0': 'train_bureau_a_1_0.csv', 'bureau1': 'train_bureau_a_1_1.csv',
    'bureau2': 'train_bureau_a_1_2.csv', 'bureau3': 'train_bureau_a_1_3.csv',
    'bureau4': 'train_bureau_a_2_0.csv', 'bureau5': 'train_bureau_a_2_1.csv',
    'bureau6': 'train_bureau_a_2_2.csv', 'bureau7': 'train_bureau_a_2_3.csv',
    'bureau8': 'train_bureau_a_2_4.csv', 'bureau9': 'train_bureau_a_2_5.csv',
    'bureau10': 'train_bureau_a_2_6.csv', 'bureau11': 'train_bureau_a_2_7.csv',
    'bureau12': 'train_bureau_a_2_8.csv', 'bureau13': 'train_bureau_a_2_9.csv',
    'bureau14': 'train_bureau_a_2_10.csv','bureau15': 'train_bureau_b_1.csv', 
    'bureau16': 'train_bureau_b_2.csv',
    'debit': 'train_debitcard', 



    }
Dss = dataSetSet()
Dss.joinDataSubsets(dataTrainNames)



