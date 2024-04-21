from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

""" Declare categorial and numerical features. Filter df down to these features and 
    case_id and target 
"""
def processMain(df, keep_col=None):
    # Read in variables from features.csv, which contains the categorical and numerical features
    f_df = pd.read_csv('features.csv')
    cat_f = f_df[f_df['type'] == 'categorical']['name'].tolist()
    num_f = f_df[f_df['type'] == 'numerical']['name'].tolist()
    col_to_keep = keep_col + cat_f + num_f
    df = df[col_to_keep]
    print(f"df columns in process Main: {df.columns}")
    df.to_csv('test/TEST0.csv', index=False)

    # For categorical variables, take the most recent value
    for col in cat_f:
        for case_id in df['case_id'].unique():
            try: 
                common_val = df[df['case_id'] == case_id][col].value_counts().idxmax()
            except ValueError: 
                common_val = 0
            df.loc[df['case_id'] == case_id, col] = common_val
            # print(f"common_val: {common_val}")

    # For numerical variables, take the mean of the values
    for col in num_f:
        for case_id in df['case_id'].unique():
            common_val = df[df['case_id'] == case_id][col].mean()
            df.loc[df['case_id'] == case_id, col] = common_val

    # round to 2 decimal places. Replace missing numerical values with the mean
    df.loc[:, num_f] = df[num_f].round(2)
    df.loc[:, num_f] = df[num_f].fillna(df[num_f].mean())

    df.to_csv('test/TEST1.csv', index=False)

    # Now for case_id's the data is aggregated, so now we can drop duplicates
    df = df.drop_duplicates(subset='case_id')
    df.to_csv('test/TEST2.csv', index=False)

    # one-hot encode the categorical variables
    df = pd.get_dummies(df, columns=cat_f, dtype=np.int8)

    # standardize the continuous variables
    scaler = StandardScaler()
    df[num_f] = scaler.fit_transform(df[num_f]) 

    return df

