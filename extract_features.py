import librosa
import numpy as np
import pandas as pd
import scipy
from scipy.stats import skew
from tqdm import tqdm, tqdm_pandas
import csv
import os
import glob
import sklearn
# tqdm.pandas()

# path = 'C:\\Users\\theco\\Desktop\\03-01-01-01-01-01-01.wav'
# path = 'C:\\Users\\theco\\Desktop\\Audio Folder\\'
# path = 'C:\\Users\\theco\\Desktop\\Test Folder\\'
SAMPLE_RATE = 16000
frames = 20


# def get_field_names():
#     field_names = ['filename']
#     for i in range(0, frames):
#         field_names.append('mfcc_feature_' + str(i))
#     field_names.append('emotion_name')
#     field_names.append('value')
#     return field_names

def get_field_names():
    field_names = ['filename']
    for i in range(0, frames):
        field_names.append('mfcc_feature_' + str(i))

    for i in range(0, 128):
        field_names.append('mel_'+str(i))

    # for i in range(0, 12):
    #     field_names.append('chroma_'+str(i))
    # for i in range(0, 7):
    #     field_names.append('contrast'+str(i))
    # for i in range(0, 6):
    #     field_names.append('tonnetz'+str(i))
    field_names.append('emotion_name')
    field_names.append('value')
    return field_names


def segregate_function(filename , mfccs, mel):
    value = filename[0:2]
    emotion_name = ''
    if value == '01':
        emotion_name = 'neutral'
    elif value == '02':
        emotion_name = 'calm'
    elif value == '03':
        emotion_name = 'happy'
    elif value == '04':
        emotion_name = 'sad'
    elif value == '05':
        emotion_name = 'angry'
    elif value == '06':
        emotion_name = 'fearful'
    elif value == '07':
        emotion_name = 'disgust'
    elif value == '08':
        emotion_name = 'surprise'

    dataset = 'dataset.csv'
    fieldnames = get_field_names()
    if not os.path.exists(dataset):
        with open(dataset, 'w', newline='') as my_csv:
            writer = csv.DictWriter(my_csv, fieldnames=fieldnames)
            writer.writeheader()
            my_csv.close()

    with open(dataset, 'a', newline='') as my_csv:
        writefile = csv.writer(my_csv)
        row = [filename]

        for i in range(0, frames):
            row.append(mfccs[i])

        for i in range(0, 128):
            row.append(mel[i])

        # for i in range(0, 12):
        #     row.append(chroma[i])
        # for i in range(0, 7):
        #     row.append(contrast[i])
        #
        # for i in range(0, 6):
        #     row.append(tonnetz[i])

        row.append(emotion_name)

        # value = int(value)-1

        row.append(value)
        writefile.writerow(row)
        my_csv.close()
        # writer.write(row)


# def mfcc(path, fieldnames):
#     sig, rate = librosa.load(path=path, sr=SAMPLE_RATE)
#     # frames = frames
#     # window_size = 512 * (frames - 1)
#     mfcc_features = librosa.feature.mfcc(sig, sr=rate)
#     # mfcc_features = librosa.feature.mfcc(y=sig[:window_size],sr=rate)
#     # new_path = path.replace('C:\\Users\\theco\\Desktop\\Audio Folder\\03-01-', '')
#     new_path = path.replace('C:\\Users\\theco\\Desktop\\Test Folder\\03-01-', '')
#     # scaler = sklearn.preprocessing.StandardScaler()
#     # mfcc_features_scaled = scaler.fit_transform(mfcc_features)
#     mean = mfcc_features.mean(axis=0)
#     # mean = mfcc_features_scaled.mean(axis=1)
#     segregate_function(new_path, mean, fieldnames)
# def mfcc(filename, fieldnames):
#     # sig, rate= librosa.load(path=filename, res_type='kaiser_fast')
#     sig, rate = librosa.load(os.path.join(subdir,filename), res_type='kaiser_fast')
#     mean = np.mean(librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=40).T, axis=0)
#     new_path = filename.replace('03-01-', '')
#     # new_path = path.replace('C:\\Users\\theco\\Desktop\\Test Folder\\03-01-', '')
#     segregate_function(new_path, mean, fieldnames)


def feature_extraction(filename):
    # sig, rate= librosa.load(path=filename, res_type='kaiser_fast')
    x, sample_rate = librosa.load(os.path.join(subdir, filename), res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=frames).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(x, sr=sample_rate).T, axis=0)
    # stft = np.abs(librosa.stft(X))
    # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    #                                           sr=sample_rate).T, axis=0)
    # new_path = path.replace('C:\\Users\\theco\\Desktop\\Test Folder\\03-01-', '')
    # segregate_function(new_path, mfccs, chroma, mel, contrast, tonnetz)
    new_path = filename.replace('03-01-', '')
    segregate_function(new_path, mfccs, mel)


path = 'C:\\Users\\theco\\Desktop\\College work\\Project\\Audio Speech Database\\'
for subdir, dirs, files in os.walk(path):
    for file_name in files:
        try:
            # print filename
            feature_extraction(file_name)
        except ValueError:
            continue


# for filename in glob.glob(os.path.join(path, '*.wav')):