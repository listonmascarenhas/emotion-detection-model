import keras
import joblib
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from keras import backend
import os


# def predictive_model(file_path):
#     # X = joblib.load(filepath + 'X.joblib')
#     # y = joblib.load(filepath + 'y.joblib')
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#     loaded_model = keras.models.load_model(
#         'C:\\Users\\theco\\Desktop\\College work\\Project\\Speech_model\\Test Product.h5')
#     # loaded_model.summary()
#     # x_testcnn = np.expand_dims(X_test, axis=2)
#     # loss, acc = loaded_model.evaluate(x_testcnn, y_test)
#     # print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#     # path = 'C:\\Users\\theco\\Desktop\\New folder\\emotion.wav'
#     sig, rate = librosa.load(path=file_path, res_type='kaiser_fast')
#     mean = np.mean(librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=40).T, axis=0)
#     mean_dim = np.expand_dims(mean, axis=2)
#     mean_dim = np.transpose(mean_dim)
#     mean_dim = np.expand_dims(mean_dim, axis=2)
#     # print(X.shape)
#     # print(X_test.shape)
#     # print(X_train.shape)
#     # print(x_testcnn.shape)
#     # print(np.transpose(mean).shape)
#     # print(mean1.shape)
#     predict = loaded_model.predict_classes(mean_dim)
#     K.clear_session()
#     return str(predict[0]), mean

def predictive_model(file_path):
    loaded_model = keras.models.load_model('Speech Recognition Model.h5')
    # loss, acc = loaded_model.evaluate(x_testcnn, y_test)
    # print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    sig, rate = librosa.load(path=file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=20).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(sig, sr=rate).T, axis=0)

    features = np.concatenate((mfccs, mel))
    expanded_features = np.expand_dims(features, axis=0)
    predict = loaded_model.predict_classes(expanded_features)
    # print(predict)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    emo = emotions[predict[0]]
    # print(emo)
    # print(predict)
    # print(predict[0])
    # print(emo)
    backend.clear_session()
    # return str(predict[0]), mean_dim
    return str(emo), features


# path = 'C:\\Users\\theco\\Desktop\\Test folder\\03-01-02-02-02-01-01.wav'
# path = 'C:\\Users\\theco\\PycharmProjects\\emotion-detection\\audiorecordtest.wav'
# print(predictive_model(path)[0])

