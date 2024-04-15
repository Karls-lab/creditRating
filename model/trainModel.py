from model import NLTK_Binary_Classifier
from sklearn.model_selection import train_test_split
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def splitTrainingData(df, featureCols, targetCol, random=False):
    state = 42 if random else None
    X = df[featureCols]
    X = X.drop(columns=['target'])
    y = df[targetCol]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    return X_train, X_test, y_train, y_test


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root, 'data', 'processed', 'final.csv')
    df = pd.read_csv(file_path)
    print("NUMber of columns: ", len(df.columns))
    print(df)
    df = pd.read_csv(os.path.join(root, 'data', 'processed', 'final.csv'))

    columns = df.columns
    X_train, X_test, y_train, y_test = splitTrainingData(df, columns, 'target')
    model = NLTK_Binary_Classifier()
    model.compile()

    # Graph the confusion matrix with seaborn
    history = model.fit(X_train, y_train, epochs=1, batch_size=64)

    conf_matrix = confusion_matrix(y_train,  model.model.predict(X_train).round()) 
        
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Loan Default')
    plt.ylabel('Actual Load Default')
    plt.title('Confusion Matrix Loan Default Prediction (0-Non-Default, 1-Default)')
    plt.show()
    
    # Save model and reset weights
    model.model.save('model/model.keras')
    model.reset_weights()




main()