import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'


model = tf.keras.models.load_model('Malaria.h5')
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(100, 100))
    x = image.img_to_array(data)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images)
    return classes


# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)


        result = api(full_name)

        if result < 0.5:
            label = 'Infected'

        else:
            label = 'Not Infected'

    return render_template('predict.html', image_file_name = file.filename, label = label)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True