from Model import B3D3AD_Classifier
from sklearn.model_selection import train_test_split
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np


def splitTrainingData(df, featureCols, targetCol, random=False):
    state = 42 if random else None
    X = df[featureCols]
    X = X.drop(columns=['target'])
    y = df[targetCol]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    return X_train, X_test, y_train, y_test


def main():
    # Load the data
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root, 'data', 'processed', 'final.csv')
    df = pd.read_csv(file_path)

    # Split features and target columns
    featureCols = df.columns.drop('target')
    X = df[featureCols]
    y = df['target']

    # Initialize the k-fold cross-validator
    k_folds = 5  # Choose the number of folds
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize lists to store evaluation results
    accuracies = []
    recalls = []

    # Perform k-fold cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model on a custom Model Class (NLTK_Binary_Classifier)
        model = B3D3AD_Classifier()
        model.fit(X_train, y_train)  # Assuming your fit method works with your custom class

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)
        recall = model.recall(X_test, y_test)
        recalls.append(recall)

    # Calculate and print the mean accuracy
    mean_accuracy = np.mean(accuracies)
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean recall: {np.mean(recalls)}")

     # Display the Confusion Matrix
    conf_matrix = confusion_matrix(y_train,  model.model.predict(X_train).round()) 
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Loan Default')
    plt.ylabel('Actual Load Default')
    plt.title('Confusion Matrix Loan Default Prediction (0-Non-Default, 1-Default)')
    plt.show()
    

if __name__ == "__main__":
    main()

main()