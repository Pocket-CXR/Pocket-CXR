import numpy as np
import torch
from fastai.vision.all import *
from werkzeug.utils import secure_filename


from flask import Flask, request, render_template
import os
app = Flask(__name__)
app.secret_key = "development-key"

UPLOAD_FOLDER = 'tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
@app.route("/homepage", methods=["GET", "POST"])
def index():
    # Home Page
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file is not None:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            export_path = Path('export.pkl')
            learner = load_learner(export_path)
            preds, idx, output = learner.predict(filepath)
            probab = output[0].item()
            probability = probab * 100
            if preds == "PNEUMONIA":
                result = "Abnormal"
            else:
                result = "Normal"
            return render_template("result.html", results=result, probability = probability)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
