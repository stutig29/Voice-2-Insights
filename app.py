import os
#import magic
import urllib.request
from appdetails import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from audio_to_text import extract_text
from summary import summarize,preprocess_text
from keyword_extraction import extract_keyword
from extract_NER import extract_entities
from spacy import displacy
import spacy
import pandas as pd
sp = sp = spacy.load('en_core_web_sm')

UPLOAD_FOLDER = '/home/stuti/Documents/PBL/voice2_insights/files_uploaded/'
FLAG = []
text_file_location = []
FLAG.append('FALSE')
ALLOWED_EXTENSIONS = set(['mp4','wav','mp3','txt'])
dbuser = pd.read_excel("/home/stuti/Documents/PBL/voice2_insights/userdata.xlsx")

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def form():
    return render_template("login.html")

@app.route("/signin/", methods = ['POST'])
def signin():
	return render_template("signin.html")

@app.route("/signup_/", methods = ['POST'])
def signup_():
	return render_template("signup.html")

@app.route("/welcome", methods = ['POST'])
def welcome():
    result = request.form
    for i in range(len(dbuser['user_name'])):
        if dbuser['user_name'][i] == result['your_name'] and dbuser['pass'][i] == result['your_pass']:
            return render_template("upload.html")
        elif i == len(dbuser['user_name']) - 1:
            return render_template("loginfail.html")
        
@app.route('/signup', methods=['POST'])
def signup():
    signup_data = request.form
    name = signup_data['name']
    password = signup_data['pass']
    email = signup_data['email']
    dict1 = {'user_name':name, 'pass' : password,'email' : email}
    
    global dbuser
    dbuser = dbuser.append(dict1,ignore_index=True)
    dbuser.to_excel("userdata.xlsx", index = False)
    return '<p>User created successfully!</p><a href="/">return</a>'

upload_file_location = []
	
'''@app.route('/')
def upload_form():
	return render_template('upload.html')'''

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)

		file = request.files['file']
		
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			upload_file_location.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			text_file_location=[]
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return render_template('dashboard.html')
		
		else:
			flash('Allowed file types are mp4,wav,mp3,txt')
			return redirect(request.url)

@app.route('/text/',methods=["POST"])
def show_text():
	file = upload_file_location[0]
	if (os.path.basename(file).split('.')[1]) == "mp4":
		text_file = extract_text(file)
		text_file_location.append(text_file)
		FLAG.append('TRUE')
		with open(text_file) as f:
			text = f.readlines()
		text = ''.join(text)
		clean_text = preprocess_text(text)
		final_text = ''.join(clean_text)
		flash(final_text)
	elif (os.path.basename(file).split('.')[1]) == "wav":
		text_file = extract_text(file)
		text_file_location.append(text_file)
		FLAG.append('TRUE')
		with open(text_file) as f:
			text = f.readlines()
		text = ''.join(text)
		clean_text = preprocess_text(text)
		final_text = ''.join(clean_text)
		flash(final_text)
	else:
		text_file = file
		text_file_location.append(text_file)
		FLAG.append('TRUE')
		with open(text_file) as f:
			text = f.readlines()
		text = ''.join(text)
		clean_text = preprocess_text(text)
		final_text = ''.join(clean_text)
		flash(final_text)
	return render_template('dashboard.html')


@app.route('/smry/',methods=["POST"])
def extract_summary():
	if len(FLAG)==1:
		file = upload_file_location[0]
		if (os.path.basename(file).split('.')[1]) == "mp4":
			text_file = extract_text(file)
		elif (os.path.basename(file).split('.')[1]) == "wav":
			text_file = extract_text(file)
		else:
			text_file = file
	else:
		text_file = text_file_location[0]
	s = []
	ranked_sentences = summarize(text_file)
	for i in range(2):
		s = ''.join(ranked_sentences[i][1])
	flash(s)
	return render_template('dashboard.html')


@app.route('/key/',methods=['POST'])
def find_keywords():
	if len(FLAG)==1:
		file = upload_file_location[0]
		if (os.path.basename(file).split('.')[1]) == "mp4":
			text_file = extract_text(file)
		elif (os.path.basename(file).split('.')[1]) == "wav":
			text_file = extract_text(file)
		else:
			text_file = file
	else:
		text_file = text_file_location[0]
	outputs = []
	words = extract_keyword(text_file)
	for w in words[:5]:
		flash(w[0])
	return render_template('dashboard.html')

@app.route('/ner/',methods=['POST'])
def find_entities():
	if len(FLAG)==1:
		file = upload_file_location[0]
		if (os.path.basename(file).split('.')[1]) == "mp4":
			text_file = extract_text(file)
		elif (os.path.basename(file).split('.')[1]) == "wav":
			text_file = extract_text(file)
		else:
			text_file = file
	else:
		text_file = text_file_location[0]

	quantity,org,norp,product,person,cardinal,ordinal,money,date = extract_entities(text_file)

	flash('ORG')
	if len(org) == 0:
		flash('NONE')
	else:
		for i in org:
			flash(i)

	
	flash('PRODUCT')
	if len(product) == 0:
		flash('NONE')
	else:
		for i in product:
			flash(i)
	

	flash('QUANTITY')
	if len(quantity) == 0:
		flash('NONE')
	else:
		for i in quantity:
			flash(i)

	return render_template('dashboard.html')


if __name__ == "__main__":
    app.run(debug=True)