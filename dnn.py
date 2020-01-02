
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from save_model import save_model

data = pd.read_csv('dataset.csv')
# new_data = data.drop('value', axis=1)
new_data = data.drop('emotion_name', axis=1)
new_data = new_data.drop('filename', axis=1)
#new_data = new_data.drop('mfcc_feature_0', axis=1)
# print(new_data.head())
# print(new_data.size)
# X = new_data.drop('emotion_name', axis=1)
X = new_data.drop('value', axis=1)
# y = new_data['emotion_name']
y = new_data['value']
# print X.head()
# print y.head()


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels+1))
    one_hot_encode[np.arange(n_labels), labels] = 1
    one_hot_encode=np.delete(one_hot_encode, 0, axis=1)
    return one_hot_encode


# train_y = one_hot_encode(train_y)
# test_y = one_hot_encode(test_y)

y = one_hot_encode(y)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
# print(train_x.shape)


n_dim = train_x.shape[1]
n_classes = train_y.shape[1]
units_1 = n_dim
units_2 = 150
units_3 = 75
units_4 = 35
# n_hidden_units_1 = n_dim
# n_hidden_units_2 = int(n_dim*2)
# n_hidden_units_3 = int(n_dim/2)
# n_hidden_units_4 = int(n_dim/4)


def create_model(activation_function='relu', init_type='normal', optimiser='adam', dropout_rate=0.2):
    model = Sequential()

    model.add(Dense(units_1, input_dim=n_dim, init=init_type, activation=activation_function))
    # model.add(Dropout(dropout_rate))

    model.add(Dense(units_2, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units_3, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units_4, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))

    model.add(Dense(n_classes, init=init_type, activation='softmax'))
   #model compilation
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    return model


model = create_model()
history = model.fit(train_x, train_y, epochs=500, batch_size=16)
predict = model.predict(test_x, batch_size=16)
emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
#predicted emotions from the test set
y_pred = np.argmax(predict, 1)
predicted_emo=[]
for i in range(0,test_y.shape[0]):
    emo=emotions[y_pred[i]]
    predicted_emo.append(emo)

actual_emo=[]

y_true=np.argmax(test_y, 1)
for i in range(0,test_y.shape[0]):
    emo = emotions[y_true[i]]
    actual_emo.append(emo)
print(confusion_matrix(actual_emo,predicted_emo))
print(classification_report(actual_emo, predicted_emo))
print("Accuracy score is %.2f " % accuracy_score(actual_emo, predicted_emo))
save_model(model)

#generate the confusion matrix
# cm =confusion_matrix(actual_emo, predicted_emo)
# index = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
# columns = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
# cm_df = pd.DataFrame(cm,index,columns)
# print(cm_df)
# cnnhistory=model.fit(train_x, train_y, batch_size=16, epochs=500, validation_data=(test_x, test_y))
# plt.plot(cnnhistory.history['loss'])
# plt.plot(cnnhistory.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# #
# plt.plot(cnnhistory.history['acc'])
# plt.plot(cnnhistory.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

