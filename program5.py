import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
names = ['sepal_length','sepal_width','petal_length','petal_width','Class']
dataframe = pd.read_csv(r'C:\Users\acer\Downloads\P5Data.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
classifier = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
ypred = classifier.predict(X_test)

print('Original label   Predicted label     Correct/Wrong')
for i, label in enumerate(y_test):
    predicted = ypred[i]
    status = 'Correct' if label == predicted else 'Wrong'
    print(f'{label:16} {predicted:16} {status}')
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, ypred))

print("\nClassification Report:")
print(metrics.classification_report(y_test, ypred))

print("Accuracy: %.2f" % metrics.accuracy_score(y_test, ypred))
