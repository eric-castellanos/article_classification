import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def main():
    train = pd.read_csv('../data/processed/train.csv')
    test = pd.read_csv('../data/processed/test.csv')

    target_classes = ["World", "Sports", "Business", "Sci/Tech"]

    X_train, X_test, y_train, y_test = train_test_split(train.loc[:, ~train.columns.isin(['Class Index', 'Title', 'Description'])], train['Class Index'], test_size=0.25, random_state=42)

    model = LogisticRegression(random_state=0, multi_class='multinomial')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    print("Confustion Matrix for Model - Train:")
    print(confusion_matrix(y_train, y_pred))

    print("Classification Report for Model - Test:")
    print(classification_report(y_train, y_pred, target_names=target_classes))

    y_pred = model.predict(X_test)

    print("Confustion Matrix for Model - Test:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report for Model - Test:")
    print(classification_report(y_test, y_pred, target_names=target_classes))

    joblib.dump(model, "models/multinomial_logistic_regression.pkl") 

if __name__ == '__main__':
    main()