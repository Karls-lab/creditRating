from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def main():
    start = time.time()

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

        # Train the model on Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        recall = recall_score(y_test, y_pred)
        recalls.append(recall)

    # Calculate and print the mean accuracy
    mean_accuracy = np.mean(accuracies)
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean Recall: {np.mean(recalls)}")

    end = time.time()
    print(f"Time taken: {end - start} seconds")


    # Display the Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Target')
    plt.ylabel('Actual Target')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
