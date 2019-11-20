from flask import Flask

UPLOAD_FOLDER = '/home/stuti/Documents/PBL/voice2_insights/files_uploaded/'
app = Flask(__name__)
app.secret_key = "b67551d6e2ac725f40b0bcf9ce340c4b"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024