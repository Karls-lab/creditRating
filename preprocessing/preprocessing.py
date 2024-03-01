import pandas as pd
import os 
import mrmr
from mrmr import mrmr_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

rootPath = os.path.dirname(os.path.dirname(__file__))
docsPath = os.path.join(rootPath, 'preprocessing', 'docs')
dataSetPath = os.path.join(
    os.path.join(rootPath, 'data', 'mergedDatasets', 'person&Applprev.csv'))

# count the number of null values in each column
def printNullValues(df, fileName):
    fileName = "Num_Null_" + fileName
    if not os.path.exists(docsPath):
        os.mkdir(docsPath)
    series = df.isnull().sum().sort_values()
    df = series.reset_index()
    df.columns = ['ColumnName', 'NumNullValues']
    df.to_csv(os.path.join(docsPath, fileName), index=False)


""" For each categorical feature, we encode it into a unique numerical value (0, 1, 2, 3, 4, 5)
    one number for each category. Then, scale the values to be between 0 and 1. (0, .25, .5, .75, 1)"""
def labelEncodeCategorical(df, categoricalFeatures):
    # TODO filter numerous bad values
    for feature in categoricalFeatures:
        df = df[df[feature] != '?']
    label_encoder = LabelEncoder()
    min_max_scaler = MinMaxScaler()
    for col in categoricalFeatures:
        df[col] = label_encoder.fit_transform(df[col])
    df[categoricalFeatures] = min_max_scaler.fit_transform(df[categoricalFeatures])
    return df



""" mRMR is an algorithm that ranks the most relevant features in a dataset."""
def printMRMRValues(df, fileName):
    fileName = "MRMR_" + fileName
    if not os.path.exists(docsPath):
        os.mkdir(docsPath)
    allFeatures = (df.columns)
    # print(df.columns)
    selectedFeatures = mrmr_classif(X=df[allFeatures], y=df['target'], K=len(allFeatures))
    # save the selected features to a file
    df = pd.DataFrame(selectedFeatures)
    df.to_csv(os.path.join(docsPath, fileName), index=False)



df = pd.read_csv(dataSetPath)
categoricalVariables = pd.read_csv(os.path.join(rootPath, 'preprocessing', 
                                'docs', 'pApplCategoricalVars.csv'))
catVarsSeries = categoricalVariables['categoricalVariables']

""" LIST OF FEATURES TO THROW AWAY"""
# one-hot encode the categorical variables
df = pd.get_dummies(df, columns=catVarsSeries)
df = df.drop(columns=['date_decision', 'birth_259D', 'birthdate_87D'])

# for values in catVarsSeries: if a value isn't numeric, print the column name
print(df.info())

printMRMRValues(df, 'pApplIndex.csv')


# df = labelEncodeCategorical(df, catVarsSeries)
# df.to_csv(os.path.join(rootPath, 'data', 'processed', 'pApplIndexTest.csv'), index=False)