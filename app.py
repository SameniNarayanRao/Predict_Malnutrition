from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("D:/ML/Child Status/randomforest.pkl", "rb"))

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def home():
    Height = request.form['a']
    Weight = request.form['b']
    array = np.array([[tenure, Height, Weight]])
    pred = model.predict(array)
    return render_template("result.html", data = pred)



if __name__ == "__main__":
    app.run(debug = True)
