#import libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report

import pickle

#load dataset
data = pd.read_csv('schistoma.csv')

#defining the data into (X) Independent and (y) dependent variables
X = data[['abdominal', 'diarrhea', 'bloody_stool', 'bloody_urine', 'swim', 'dam_river_use', 'urinating_stool_in_water', 'boil_water_use']]
y = data['status'] 

#spliting the into training and test based on 80/20%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

#machine Learning algorithms
logic = LogisticRegression()
vector = SVC()
bayes = GaussianNB()

#training dataset using machine learning algorithms
logic.fit(X_train, y_train)
vector.fit(X_train, y_train)
bayes.fit(X_train, y_train)

#confidence score counts with test data
logic_score = logic.score(X_test, y_test)
vector_score = vector.score(X_test, y_test)
bayes_score = bayes.score(X_test, y_test)

print('---------Prediction Scores------')

print('Logistic prediction confidence is', logic_score*100,'%')
print('Vector prediction confidence is', vector_score*100,'%')
print('Bayes prediction confidence is', bayes_score*100,'%')


#prediction test
test = [[1,1,0,1,0,0,1,1]]

logic_test = logic.predict(test)
vector_test = vector.predict(test)
bayes_test = bayes.predict(test)

print('---------Test Scores------')

print("logistic = ", logic_test)
print("vector = ", vector_test)
print("bayes =", bayes_test)

#evaluate algorithms performances using confusion matrix/classification report
y_log = logic.predict(X_test)
y_vec = vector.predict(X_test)
y_bay = bayes.predict(X_test)

print('---------Classification Report------')

print('Logistic Regression Classfier Report:')
print(classification_report(y_test, y_log))

print('Support Vector Classifier Report:')
print(classification_report(y_test, y_vec))

print('Naive Bayes Classifier Report:')
print(classification_report(y_test, y_bay))


with open('model', 'wb') as f:
    pickle.dump(logic, f)


with open('model', 'rb') as f:
    model = pickle.load(f)


test = [[1,1,0,1,0,0,1,1]]
result2 = model.predict(test)
print(result2)
