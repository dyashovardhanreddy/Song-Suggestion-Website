from __future__ import division, print_function
from flask import Flask, redirect , url_for,Response ,render_template,request
from flask_sqlalchemy import SQLAlchemy
import enum
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2
from imutils.video import WebcamVideoStream

import sys
import os
import glob
import re
import numpy as np


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///site.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False



#loading saved model 

json_file = open('./model/model.json','r')
loaded_model_json =json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model/model.h5")
cam = cv2.VideoCapture(0)
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#database
db = SQLAlchemy(app)


#Song model to save song according to there category

class Song(db.Model):
	_id = db.Column("id", db.Integer,primary_key=True)
	category = db.Column("category",db.String(8),nullable=False)
	name = db.Column("name",db.String(100),nullable=False)
	link = db.Column("link",db.String(500),nullable=False) 
	
	def __init__(self,name,link,category):
		 self.name = name
		 self.link = link
		 self.category = category

#home page

@app.route('/',methods=['POST','GET'])
@app.route('/<action>',methods=['POST','GET'])
def home(action=None):
	cam = cv2.VideoCapture(0)
	if action is None:
		return render_template("index.html",src="../static/images/listening-songs.jpg",start_btn="",stop_btn="disable",predict_btn="disable")
	elif action == 'stop':
		return render_template("index.html",src="../static/images/listening-songs.jpg",start_btn="",stop_btn="disable",predict_btn="disable")
	elif action == 'predict':
		save_image()
		return prediction()
	else:
		return render_template("index.html",src=url_for('video'),start_btn="disable",stop_btn="",predict_btn="")
def save_image():
	while True:
		success,frame=cam.read()
		if not success:
			continue
		gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces_detected = face_haar_cascade.detectMultiScale(gray_img,1.32,5)

		for (x,y,w,h) in faces_detected:
			original=frame
			cv2.imwrite('./static/images/original.jpg',original)
			img=gray_img[y:y+w,x:x+h]
			cv2.imwrite('./static/images/img.jpg',img)
		return

def generate_frames():
    while True:
        success,frame=cam.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        	


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


#prediction 




@app.route("/prediction")
def prediction():


	img = cv2.imread('./static/images/img.jpg',0)

	img = cv2.resize(img ,(48,48))

	img = np.reshape(img, (1,48,48,1))

	predictions= loaded_model.predict(img)

	max_index = np.argmax(predictions[0])

	emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

	predicted_emotion = emotions[max_index]

	

	print(predicted_emotion)	

	return redirect(url_for('songs', emotion = predicted_emotion))


# songs page

@app.route('/songs',methods=['GET','POST']) #when user wants all types of songs
@app.route('/songs/<emotion>', methods=['GET', 'POST']) #when user wants to here songs according to his mood
def songs(emotion=None):
	headings=("Category","Name","Link")
	if emotion is None:
		return render_template("songs.html",table_header=headings,songs=Song.query.all())
	else:
		return render_template("songs.html",table_header=headings,emotion=emotion,songs=Song.query.filter_by(category=emotion))



if __name__ == '__main__':
	db.create_all()
	app.run(debug=True)
#by:Yasho Vardhan
