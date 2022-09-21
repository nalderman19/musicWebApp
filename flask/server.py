"""

client -> POST request -> server extract/predict -> prediction -> client

"""
import os
from random import randint as random
from flask import Flask, request, jsonify

from keyword_spotting_service import Keyword_Spotting_Service

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random(0,100000))
    audio_file.save(file_name)
    
    # invoke keyword spotting service
    kss = Keyword_Spotting_Service()
    
    # make prediction
    prediction = kss.predict(file_name)
    
    # remove the audio file
    os.remove(file_name)
    
    # send back prediction in json
    data = {"keyword": prediction}
    
    
    return jsonify(data)


if __name__ == "__main__":
    os.environ ['no_proxy'] = '*'
    app.run(debug=False)