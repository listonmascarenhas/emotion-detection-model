# import inline as inline
# import matplotlib
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from save_model import save_model

data = pd.read_csv('dataset.csv')
new_data = data.drop('value', axis=1)
new_data = new_data.drop('filename', axis=1)
#new_data = new_data.drop('mfcc_feature_0', axis=1)
X = new_data.drop('emotion_name', axis=1)
y = new_data['emotion_name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print X.head()
# print y.head()

# svm_classifier = SVC(kernel='rbf')
svm_classifier = SVC(kernel='poly', degree=8)
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy score is %.2f " % accuracy_score(y_test, y_pred))


# #training
# train_data = pd.read_csv('training_dataset.csv')
# new_train_data = train_data.drop('value', axis=1)
# new_train_data = new_train_data.drop('filename', axis=1)
# X_train = new_train_data.drop('emotion_name', axis=1)
# y_train = new_train_data['emotion_name']
# #testing
# test_data = pd.read_csv('testing_dataset.csv')
# new_test_data = test_data.drop('value', axis=1)
# new_test_data = new_test_data.drop('filename', axis=1)
# X_test = new_test_data.drop('emotion_name', axis=1)
# y_test = new_test_data['emotion_name']
# svm_classifier = SVC(kernel='poly', degree=8)
