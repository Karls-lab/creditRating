import os 
import pandas as pd
import numpy as np


""" Takes a number of datasets, and merges them int one dataset. 
    It also specifies the number of rows the dataset should have.
    This is part of our data exploration process.
"""
class dataSetSet():
    def __init__(self):
        self.filePath = os.path.dirname(__file__)
        self.dataFolder = os.path.join(os.path.dirname(self.filePath), 'data')
        self.subsetFolder = os.path.join(self.dataFolder, 'subset')
        self.mergedFolder = os.path.join(self.dataFolder, 'mergedDatasets')
        self.creditFolder = "home-credit-credit-risk-model-stability"

    def getTrainData(self):
        return os.path.join(self.dataFolder, self.creditFolder, 'csv_files', 'train')


    """
    Opens the dataframe, and takes a number of rows denoted 'n'
    and saves it to a new file in the subset folder.
    filename is train_base.csv, train_applprev_1_0.csv, ...
    SAMPLE: dataSubset(n=100, filename='train_base.csv', random_state=42)
    """
    def dataSubset(self, n, filename):
        if os.path.exists(os.path.join(self.subsetFolder, filename)):
            print(f'{filename} already exists in subset folder')
            return
        print(f'Creating subset of {filename} with {n} rows')
        df = pd.read_csv(os.path.join(self.getTrainData(), filename))
        # df = df.sample(n=n, random_state=random_state)
        df = df[0:n]
        if not os.path.exists(self.subsetFolder):
            os.makedirs(self.subsetFolder)
        df.to_csv(os.path.join(self.subsetFolder, filename), index=False)


    """
    Opens two files in subset folder, and merges the data frames 
    on the primary key of 'case_id' 
    """
    def joinDataSubsets(self, saveFileName, *csvFiles):
        # Join the data subsets into one file
        # First, get the train_base.csv file
        train_basePath = os.path.join(self.subsetFolder, 'train_base.csv')
        if not os.path.exists(train_basePath):
            print('train_base.csv does not exist in subset folder')
            print('Creating one automatically...')
            self.dataSubset(n=100, filename='train_base.csv')

        # Now for each file, merge it with the train_base.csv file
        train_base_df = pd.read_csv(train_basePath)
        resultDf = train_base_df
        for filename in csvFiles:
            df = pd.read_csv(os.path.join(self.subsetFolder, filename))
            resultDf = resultDf.merge(df, on='case_id')
            print(f'\nMerged DF: {resultDf.head()}')

        # Save the merged dataframe to the mergedDatasets folder
        if not os.path.exists(self.mergedFolder):
            os.makedirs(self.mergedFolder)
        resultDf.to_csv(os.path.join(self.mergedFolder, saveFileName), index=False)


""" Directory of the training data files. """
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
    'debit': 'train_debitcard_1.csv', 
    'deposit': 'train_deposit_1.csv', 
    'other': 'train_other_1.csv',
    'person': 'train_person_1.csv',
    'static0': 'train_static_0_0.csv', 'static1': 'train_static_0_1.csv',
    'static3': 'train_static_cb_0.csv',
    'tax0': 'train_tax_registry_a_1.csv', 'tax1': 'train_tax_registry_b_1.csv',
    'tax2': 'train_tax_registry_c_1.csv'
    }


""" This is the Main function, please tell me if this doesn't work!"""
Dss = dataSetSet()
Dss.dataSubset(n=100, filename=dataTrainNames['person'])
Dss.dataSubset(n=100, filename=dataTrainNames['aplprev0'])

Dss.joinDataSubsets(
   'person&Applprev.csv', dataTrainNames['person'], dataTrainNames['aplprev0']
)



