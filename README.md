# emotion-detection-model
The model takes in an audio file as an input and outputs the emotion in a JSON format. The model is created using deep neural networks 
and is accessed using a Web API. The project is built in Python and the Web API is built using Flask.


The audio database used is The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) and can be downloaded 
from https://zenodo.org/record/1188976#.Xeu8_kf7RPY

extract_features.py
In extract_features, we extract certain features from the audio files and along with their emotion values save them to a dataset. The 
features extracted are Mel-frequency cepstrum(MFCC) and Mel-scaled spectrograms.

dnn.py
In dnn, we use Deep Neural Networks as our classifier. Since we get the highest accuracy with this model, we then save the model 
to be used later on. 

knn.py
In knn, we use K-Nearest Neighbours as our classifier. We don't achieve a good accuracy so we don't use this model.

svm_model.py
In svm_model, we use Support Vector Machines as our classifier. We don't achieve a good accuracy so we don't use this model.

__init__.py
In __init__, we create our Web APIs using Flask. 
