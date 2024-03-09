from model import NLTK_Binary_Classifier
from sklearn.model_selection import train_test_split
import os 
import pandas as pd


def splitTrainingData(df, featureCols, targetCol, random=False):
    state = 42 if random else None
    X = df[featureCols]
    y = df[targetCol]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    return X_train, X_test, y_train, y_test


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(root)
    df = pd.read_csv(os.path.join(root, 'data', 'processed', 'fiveFeatures.csv'))

    columns = df.columns
    X_train, X_test, y_train, y_test = splitTrainingData(df, columns[1:], 'target')
    model = NLTK_Binary_Classifier()
    model.compile()

    # print(X_train.head())
    # print(y_train.head())
    history = model.fit(X_train, y_train, epochs=6, batch_size=64)
    model.reset_weights()


main()