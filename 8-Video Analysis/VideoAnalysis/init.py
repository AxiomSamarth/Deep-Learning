from flask import render_template,request,url_for,redirect,session
from flask import flash
from flask import Flask
import os
import scp
import time

time.sleep(10)

app = Flask(__name__)
app.secret_key = "secret_key"

@app.route('/dashboard/', methods=['GET','POST'])
def dashboard():
	
	clip_name = request.form['clip']
	#clip_name = 'ntr_fight'
	result = scp.predict('static/test_videos/'+clip_name+'.mp4')
	return render_template('dashboard.html', value='./test_videos/'+clip_name+'.mp4', result=result)

	#return redirect(url_for('index'))

@app.route('/', methods = ['GET','POST'])
def index():
	return render_template('login.html')

if __name__ == '__main__':
	app.debug = True
	app.run()
