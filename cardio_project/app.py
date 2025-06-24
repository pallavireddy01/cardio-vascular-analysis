from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("cardio_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        int(request.form['age']),
        int(request.form['gender']),
        int(request.form['height']),
        float(request.form['weight']),
        int(request.form['ap_hi']),
        int(request.form['ap_lo']),
        int(request.form['cholesterol']),
        int(request.form['gluc']),
        int(request.form['smoke']),
        int(request.form['alco']),
        int(request.form['active']),
    ]
    
    prediction = model.predict([data])
    result = "At Risk" if prediction[0] == 1 else "Not at Risk"
    return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
