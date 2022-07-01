import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

application = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features).astype(int)

    output = round(prediction[0], 0)

    return render_template('index.html', prediction_text='Patrons Traffic is: {}'.format(output))

if __name__ == "__main__":
    application.run(debug=False)