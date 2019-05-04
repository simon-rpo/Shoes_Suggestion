import os

from flask import redirect, request, url_for, jsonify
from flask_api import FlaskAPI, exceptions, status
from werkzeug.utils import secure_filename

import settings
import prediction
import buySuggestion

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = FlaskAPI(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def response(data, uri):
    return {
        'url': uri + url_for('upload_file'),
        'data': {
            'predictions': [float(idx) for idx in data["predictions"]],
            'classes': int(data["classes"]),
            'max_prob': float(data["max_prob"])
        }
    }


def responseSugg(data, uri):
    return {
        'url': uri + url_for('upload_suggest'),
        'data': {
            'predictions': [float(idx) for idx in data["i"]["predictions"]],
            'classes': int(data["i"]["classes"]),
            'max_prob': float(data["i"]["max_prob"]),
            'suggestions': [
                [{
                    "title": idx["title"],
                    "link": idx["link"],
                    "displayLink": idx["displayLink"]
                } for idx in data["suggestions"]]
            ]
        }
    }


@app.route('/api/shoes/prediction', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filePath)
                preds = prediction.predict(filePath)
            except Exception as e:
                return e

            return response(
                preds,
                request.host_url.rstrip('/api/shoes/prediction'))
    return ''


@app.route('/api/shoes/suggestion', methods=['GET', 'POST'])
def upload_suggest():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filePath)
                preds = buySuggestion.suggest(filePath)
            except Exception as e:
                return e

            return responseSugg(
                preds,
                request.host_url.rstrip('/api/shoes/suggestion'))
    return ''


@app.route('/api/shoes/classes', methods=['GET'])
def classes_dict():
    if request.method == 'GET':
        return {
            'url': request.host_url.rstrip('/api/shoes/classes') + url_for('classes_dict'),
            'data': prediction.getClassesDict()
        }
    return ''


if __name__ == "__main__":
    app.run()
