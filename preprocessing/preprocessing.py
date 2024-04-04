import pandas as pd
import sys
import os 
import mrmr
from mrmr import mrmr_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

rootPath = os.path.dirname(os.path.dirname(__file__))
docsPath = os.path.join(rootPath, 'preprocessing', 'docs')

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


def combineCases(df):
    """
    Combine the cases into one entry. 
    Take the most common boolean feature  and the mean of the continuous features.
    """
    aggregated_df = pd.DataFrame(columns=df.columns)
    grouped = df.groupby('case_id')
    # for values in each unique case_id
    for case, case_df in grouped:
        bool_series = case_df.select_dtypes(include=['bool']).mode().iloc[0]
        numeric_series = case_df.select_dtypes(include=['int64', 'float64']).mean()
        total_values_case = pd.concat([bool_series.astype(bool), numeric_series.astype(np.float16)])
        aggregated_df.loc[case] = total_values_case 
    
    return aggregated_df


def downsample_majority_class(df, target_column, random_state=None):
    """
    Downsample the majority class in a DataFrame with imbalanced classes.
    """
    majority_class = df[df[target_column] == 0]
    minority_class = df[df[target_column] == 1]
    print("number of minority class: ", len(minority_class))
    downsampled_majority = majority_class.sample(n=len(minority_class), random_state=random_state)
    downsampled_df = pd.concat([downsampled_majority, minority_class], ignore_index=True)
    return downsampled_df


def processMain(df):
    features = pd.read_csv(os.path.join(rootPath, 'preprocessing', 'features.csv'))
    categorical_features = features[features['type'] == 'categorical']['featureName']
    # one-hot encode the categorical variables
    df = pd.get_dummies(df, df[categorical_features])
    print(df.head())
    sys.exit()
    df = pd.get_dummies(df, columns=['district_544M', 'mainoccupationinc_437A', 'cancelreason_3545846M',])
    # standardize the continuous variables
    df = normalizeColumns(df, ['annuity_853A', 'credamount_590A'])
    # fill null values with the mean of the column
    df.fillna(df.mean(), inplace=True)
    # combine the cases so each case_id is unique
    df = combineCases(df)
    return df


df = pd.read_csv(os.path.join(os.path.join(rootPath, 'data', 'mergedDatasets', 'person&Applprev.csv')))
df = df.drop_duplicates(subset=['case_id'])

# df = df[df['target'] == 1]
# df.to_csv(os.path.join(rootPath, 'data', 'processed', 'target_1.csv'), index=False)

df = processMain(df)
df = downsample_majority_class(df, 'target')
print(df['target'].value_counts())
df.to_csv(os.path.join(rootPath, 'data', 'processed', 'downsampled.csv'), index=False)
