from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    img = image.load_img(image_path, target_size=(120, 120), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    json_file = open("./models/affect_model.json", "r")

    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model_v = model_from_json(loaded_model_json)
    loaded_model_a = model_from_json(loaded_model_json)

    loaded_model_v.load_weights("./models/valence-weights-improvement-137-0.23.h5")
    loaded_model_a.load_weights("./models/arousal-weights-improvement-95-0.17.h5")

    loaded_model_v.compile(loss='mean_squared_error', optimizer='sgd')
    loaded_model_a.compile(loss='mean_squared_error', optimizer='sgd')

    prediction_v = loaded_model_v.predict(img_array)
    prediction_a = loaded_model_a.predict(img_array)
    
    str_prediction_v = str(prediction_v[0][0])  # Assuming the value is at [0][0] in the array
    str_prediction_a = str(prediction_a[0][0])  # Assuming the value is at [0][0] in the array

    classification = f"Valence: {str_prediction_v} and Arousal: {str_prediction_a}"

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)