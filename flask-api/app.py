import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, send_file

from model import Detector

app = Flask(__name__)
detector = Detector()


def run_inference(img_path = 'data/temp.jpg'):
	result_img = detector.inference(img_path)
	return result_img


@app.route("/predict", methods=['POST'])
def upload():
    file = Image.open(request.files['file'].stream)
    file.save('data/temp.jpg')
    result_img = run_inference('data/temp.jpg')
    file_object = io.BytesIO()
    result_img.save(file_object, 'PNG')   
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpg')


if __name__ == '__main__':
    app.run()



