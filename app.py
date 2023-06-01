import tensorflow as tf
import os
import numpy as np

from keras.models import load_model
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

IMAGE_SIZE = (128, 128)
MODEL_PATH = "D:/Academics/Level 6/Final Project/Model/Coconut Maturity-20230429T074521Z-001/Coconut Maturity/API/best_model_coconut_maturity_v2_6_FINAL.h5"
THRESHOLD = 0.7

class_labels = ['Immatured_Bunch', 'Immatured_Singles', 'Matured_Bunch', 'Matured_Singles']

model = load_model(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def upload():
    f = request.files['image']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    img = tf.keras.utils.load_img(file_path, target_size=IMAGE_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    print("preds: ",preds)
    print(np.max(preds))
    if THRESHOLD > np.max(preds):
        return jsonify({'error' : 'Invalid Image'})
    else:
        print(np.argmax(preds,axis=1))
        pred = class_labels[int(np.argmax(preds,axis=1))]
    return jsonify({'class' : pred})


if __name__ == '__main__':
    app.run(debug=True)
