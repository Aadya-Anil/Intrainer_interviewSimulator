#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import threading
import re
import os
from collections import Counter
import whisper
import altair as alt
### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response, render_template_string
import torch
import pyaudio
import wave
import re
import sqlite3
import matplotlib.pyplot as plt
plt.switch_backend('agg')

### Video imports ###
from library.video_emotion_recognition import *

import azure.cognitiveservices.speech as speechsdk
import pdfkit


#gramformer import
from gramformer import Gramformer




# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'


# importing the package  
import language_tool_python  
# using the tool  
my_tool = language_tool_python.LanguageTool('en-US')  



#index#

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')



################################################################################
############################### VIDEO INTERVIEW ################################
################################################################################

# Read the overall dataframe before the user starts to add his own data
df = pd.read_csv('static/js/db/histo.txt', sep=",")

# Video interview template
@app.route('/video', methods=['POST'])
def video() :
    return render_template('video.html')


@app.route('/question_1',methods = ['GET', 'POST'])
def question_1():
    # video_dash()
    return render_template('q1.html')


@app.route('/question_2',methods = ['GET', 'POST'])
def question_2():
    # video_dash()
    return render_template('q2.html')


@app.route('/question_3',methods = ['GET', 'POST'])
def question_3():
    # video_dash()
    return render_template('q3.html')



# Display the video flow (face, landmarks, emotion)
@app.route('/video_feed')
def video_feed() :
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream',methods = ['GET', 'POST'])
def stream() :
    return render_template('stream.html')


# Dashboard
@app.route('/video_dash', methods=["POST", "GET"])
def video_dash():
    df_2 = pd.read_csv('static/db/histo_perso.txt')

    def emo_prop(df_2) :
        return [int(100*len(df_2[df_2.density==0])/len(df_2)),
                    int(100*len(df_2[df_2.density==1])/len(df_2)),
                    int(100*len(df_2[df_2.density==2])/len(df_2)),
                    int(100*len(df_2[df_2.density==3])/len(df_2)),
                    int(100*len(df_2[df_2.density==4])/len(df_2)),
                    int(100*len(df_2[df_2.density==5])/len(df_2)),
                    int(100*len(df_2[df_2.density==6])/len(df_2))]

    emotions = ["Angry", "Disgust", "Fear",  "Happy", "Sad", "Surprise", "Neutral"]
    emo_perso = {}
    emo_glob = {}

    for i in range(len(emotions)) :
        emo_perso[emotions[i]] = len(df_2[df_2.density==i])
        emo_glob[emotions[i]] = len(df[df.density==i])

    df_perso = pd.DataFrame.from_dict(emo_perso, orient='index')
    df_perso = df_perso.reset_index()
    print(df_perso)
    df_perso.columns = ['EMOTION', 'VALUE']
    df_perso.to_csv('static/db/hist_vid_perso.txt', sep=",", index=False)

    df_glob = pd.DataFrame.from_dict(emo_glob, orient='index')
    df_glob = df_glob.reset_index()
    df_glob.columns = ['EMOTION', 'VALUE']
    df_glob.to_csv('static/db/hist_vid_glob.txt', sep=",", index=False)

    emotion = df_2.density.mode()[0]
    emotion_other = df.density.mode()[0]

    def emotion_label(emotion) :
        if emotion == 0 :
            return "Angry"
        elif emotion == 1 :
            return "Disgust"
        elif emotion == 2 :
            return "Fear"
        elif emotion == 3 :
            return "Happy"
        elif emotion == 4 :
            return "Sad"
        elif emotion == 5 :
            return "Surprise"
        else :
            return "Neutral"


    df_altair = pd.read_csv('static/db/gb_prob.csv', header=None, index_col=None).reset_index()
    df_altair.columns = ['Time', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    
    angry = alt.Chart(df_altair).mark_line(color='orange', strokeWidth=2).encode(
       x='Time:Q',
       y='Angry:Q',
       tooltip=["Angry"]
    )

    disgust = alt.Chart(df_altair).mark_line(color='red', strokeWidth=2).encode(
        x='Time:Q',
        y='Disgust:Q',
        tooltip=["Disgust"])


    fear = alt.Chart(df_altair).mark_line(color='green', strokeWidth=2).encode(
        x='Time:Q',
        y='Fear:Q',
        tooltip=["Fear"])


    happy = alt.Chart(df_altair).mark_line(color='blue', strokeWidth=2).encode(
        x='Time:Q',
        y='Happy:Q',
        tooltip=["Happy"])


    sad = alt.Chart(df_altair).mark_line(color='black', strokeWidth=2).encode(
        x='Time:Q',
        y='Sad:Q',
        tooltip=["Sad"])


    surprise = alt.Chart(df_altair).mark_line(color='pink', strokeWidth=2).encode(
        x='Time:Q',
        y='Surprise:Q',
        tooltip=["Surprise"])


    neutral = alt.Chart(df_altair).mark_line(color='brown', strokeWidth=2).encode(
        x='Time:Q',
        y='Neutral:Q',
        tooltip=["Neutral"])


    chart = (angry + disgust + fear + happy + sad + surprise + neutral).properties(
    width=1000, height=400, title='Probability of each emotion over time')

    chart.save('static/CSS/chart.html')
    prob = emo_prop(df_2)
    
    plot_thread = threading.Thread(target=generate_bar_plot, args=(prob, 'static/temp/plot.png'))
    plot_thread.start()

    print(df_2)
    return render_template('video_dash.html', emo=emotion_label(emotion), emo_other = emotion_label(emotion_other), prob = emo_prop(df_2), prob_other = emo_prop(df))

def generate_bar_plot(data, plot_file):
    emotions =['angry', 'disgust' , 'fear' , 'happy' , 'sad' , 'surprise' , 'neutral']
    plt.bar(emotions, data)
    plt.xlabel('emotions')
    plt.ylabel('Values')
    plt.title('Bar Plot')
    plt.savefig(plot_file)
    plt.close()

################################################################################
############################### AUDIO INTERVIEW ################################
################################################################################

def voice_recording(filename, duration=5, sample_rate=16000, chunk=1024, channels=1):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
        frames = []
        print('* Start Recording *')
        stream.start_stream()
        start_time = time.time()
        current_time = time.time()
        while (current_time - start_time) < duration:
            data = stream.read(chunk)
            frames.append(data)
            current_time = time.time()
        stream.stop_stream()
        stream.close()
        p.terminate()
        print('* End Recording * ')
        wf = wave.open(filename, 'w')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()



def enhance(text):
 
    gf=Gramformer(models=1)
    corrected_text=gf.correct(text)
    text_list=list(corrected_text)
    my_NewText=text_list[0]
    print(my_NewText)
    return my_NewText
    # getting the matches  
    # defining some variables  
    '''my_matches = my_tool.check(text)  
    myMistakes = []  
    myCorrections = []  
    startPositions = []  
    endPositions = []  
    
    # using the for-loop  
    for rules in my_matches:  
        if len(rules.replacements) > 0:  
            startPositions.append(rules.offset)  
            endPositions.append(rules.errorLength + rules.offset)  
            myMistakes.append(text[rules.offset : rules.errorLength + rules.offset])  
            myCorrections.append(rules.replacements[0])  
    
    # creating new object  
    my_NewText = list(text)   
    
    # rewriting the correct passage  
    for n in range(len(startPositions)):  
        for i in range(len(text)):  
            my_NewText[startPositions[n]] = myCorrections[n]  
            if (i > startPositions[n] and i < endPositions[n]):  
                my_NewText[i] = ""  
    
    my_NewText = "".join(my_NewText) '''
    

# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():
    # Flash message
    flash("After pressing the button above, you will have 15sec to answer the question.")
    
    return render_template('audio.html', display_button=False)

# Audio Recording
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Voice Recording
    rec_duration = 16 # in sec
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')
    voice_recording(rec_sub_dir, duration=rec_duration)

    # Send Flash message
    flash("The recording is over! Click below to view the text and grammar or click above to record again.")

    return render_template('audio.html', display_button=True)





@app.route('/audio_gram00', methods=("POST", "GET"))
def audio_gram00():
    model = whisper.load_model("base")
    result = model.transcribe("tmp/voice_recording.wav")
    print(result["text"])
    return render_template('audio_gram.html', left_side = result["text"])


@app.route('/audio_gram0', methods=("POST", "GET"))
def audio_gram0():
    input_text = request.form.get('input_text')
    results = enhance(input_text)
    return render_template('audio_gram.html', enhanced=results, left_side = input_text)






#creates a database intitially
def create_database():
    conn = sqlite3.connect('database0.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, fullname TEXT, phone_number TEXT, email TEXT, password TEXT ,address TEXT, bio_data TEXT,gender INTEGER, login INTEGER)')
    conn.commit()
    conn.close()
    return 'Database Table Created'



def add_user_data(unqid1,name,phnum,email,password,address_,gender,bio):
    
    conn = sqlite3.connect('database0.db')
    cursor = conn.cursor()
    login_ = 0
    params = (unqid1,name,phnum,email,password,address_,gender,bio,login_)
    cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", params)
    conn.commit()
    conn.close()
    return 'User Data is Inserted in Database'


# Authenticator to check if the user already logged in or not .
def authen(email, password):
    conn = sqlite3.connect('database0.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    res = cursor.fetchone()
    conn.close()
    return res



# Authenticate helper function to login and logout
def auth_helper(val,hel):
    if hel == 'login':
        conn = sqlite3.connect('database0.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET login = 1 WHERE id = ?",(val,))
        conn.commit()
        conn.close()
    elif hel == 'logout':
        conn = sqlite3.connect('database0.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET login = 0 WHERE id = ?",(val,))
        conn.commit()
        conn.close()




@app.route('/register')
def register():
    return render_template('register.html')




@app.route('/send_data', methods = ['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.files['profile_pic']
        unqid1 = request.form['unqid']
        name = request.form['firstname'] +' '+ request.form['lastname']
        email = request.form['email']
        password = request.form['password']
        phnum = request.form['phone']
        add_ = request.form['address']
        gender = request.form['gender']
        bio = request.form['bio']
        img_name = phnum
        file.save('static/images/prof_pics/'+img_name+ '.png')
    print(add_user_data(unqid1,name,phnum,email,password,add_,gender,bio))
    return render_template('success.html')



@app.route("/profile", methods = ['GET', 'POST'])
def profile():
    email = request.form['email']
    password = request.form['password']
    auth_res = authen(email,password)
    print(email)
    print(password)
    print(auth_res)
    conn = sqlite3.connect('database0.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    (unq_id,name,phnum,email,password,address,gender,bio,login) = cursor.fetchone()
    if gender==0:
        gend = 'Male'
    else:
        gend = 'Female'
    conn.close()
    return render_template('profile_actual.html',unqid = unq_id,img_name = phnum+'.png',user_name = name,bio_data = bio,address = address,full_name=name,email = email, phnum = phnum, finame=str(unq_id))

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/interview', methods = ['GET', 'POST'])
def interview():
    return render_template('index.html')


import datetime

@app.route('/file_save',methods=['POST','GET'])
def file_Save():
    print(request.files['pdfFile'])
    if 'pdfFile' in request.files:
        pdf_file = request.files['pdfFile']
        filename = pdf_file.filename
        print(filename)

        current_datetime = datetime.datetime.now()

        # Format the datetime as a string
        formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

        save_path = 'static/reports/'
        file_path = os.path.join(save_path, filename)
        pdf_file.save(file_path)


        big_filename = formatted_datetime+ '_' +filename
        big_save_path = 'static/reports/all/'
        big_file_path = os.path.join(big_save_path, big_filename)
        pdf_file.save(big_file_path)
        # Process the saved file as needed
        return 'File uploaded successfully'
    else:
        return 'No file uploaded'



@app.route('/report')
def report():
    return render_template('report.html')



if __name__ == '__main__':
    app.run(debug=True)
