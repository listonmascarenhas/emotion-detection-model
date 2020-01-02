# from flask import send_file,send_from_directory
from flask import Flask, request, jsonify, send_from_directory,render_template
from werkzeug import secure_filename
# import keras
import librosa
import numpy as np
import os
from predictive_model import predictive_model
import csv
import time
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
# import apscheduler.scheduler.Scheduler as Scheduler
app = Flask(__name__)


# def print_date_time():
#     print(time.strftime("%A, %d. %B %Y %I:%M:%S %p"))
#
#
# # scheduler = BackgroundScheduler()
# scheduler = Scheduler()
# scheduler.add_cron_job()
# # scheduler.add_job(func=print_date_time, trigger="interval", seconds=3)
#
# scheduler.start()
# # Shut down the scheduler when exiting the app
# atexit.register(lambda: scheduler.shutdown())
# #

def convert_emotion_to_no(value):
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    emo = emotions.index(value)
    return int(emo) + 1


@app.route('/')
def index():
    return "Root for predictive model."


@app.route('/upload/')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        file_name = f.filename
        file_path = os.getcwd() + '\\'+file_name
        emotion, features = predictive_model(file_path)
        data = {'emotion': emotion, 'file_name': file_name[:-4]}
        # print(convert_emotion_to_no(emotion))
        store_features = file_name[:-4] + '_features.txt'
        f = open(store_features, "w+")
        # print(mean.shape)
        for i in range(0, features.size):
            f.write(str(features[i])+'\n')
        f.close()
        return jsonify(data)


@app.route('/feedback')
def feedback():
    emotion = request.args.get('emotion')
    filename = request.args.get('filename')
    row = [filename]
    file = filename + '_features.txt'
    open_file = open(file, 'r')
    file_lines = open_file.readlines()
    i = 0
    for line in file_lines:
        i = i+1
        row.append(line.rstrip())
    print(i)
    print(row)
    row.append(emotion)
    row.append(convert_emotion_to_no(emotion))
    dataset = 'dataset.csv'
    with open(dataset, 'a', newline='') as my_csv:
        writefile = csv.writer(my_csv)
        writefile.writerow(row)
        my_csv.close()
    return 'File name is ' + filename + ' and emotion is ' + emotion


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
    # app.run(host='0.0.0.0')

# #get
# @app.route('/users/<user>')
# def hello_user(user):
#     return 'Hello %s ' % user
#
#
# @app.route('/test/')
# def test():
#     return 'Hello'
#
# @app.route('/api/post_some_data', methods= ['POST'])
# def get_text_prediction():
#     json = request.get_json()
#     print(json)
#     if len(json['text']) == 0:
#         return jsonify({'error': 'invalid input'})
#
#     return jsonify({'text': json['text']})
