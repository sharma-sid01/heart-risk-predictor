from application import app
from flask import render_template, request, json, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import requests
import numpy
import pandas as pd

#decorator to access the app
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

#decorator to access the service
@app.route("/riskclassify", methods=['GET', 'POST'])
def riskclassify():

    #extract form inputs
    age = request.form.get("age")
    gender = request.form.get("gender")
    height = request.form.get("height")
    weight = request.form.get("weight")
    ap_hi = request.form.get("ap_hi")
    ap_lo = request.form.get("ap_lo")
    cholesterol = request.form.get("cholesterol")
    gluc = request.form.get("gluc")
    smoke = request.form.get("smoke")
    alco = request.form.get("alco")
    active = request.form.get("active")

   #convert data to json
    input_data = json.dumps({"age": age, "gender": gender, "height": height, "weight": weight, "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol, "gluc": gluc, "smoke": smoke, "alco": alco, "active": active})

    #url for heart disease risk prediction model
    # url = "http://localhost:5000/api"
    url = "https://heart-risk-model-81b316a8f36a.herokuapp.com/api"
  
    #post data to url
    results =  requests.post(url, input_data)

    #send input values and prediction result to index.html for display
    return render_template("index.html", age = age, gender = gender, height = height, weight = weight, ap_hi = ap_hi, ap_lo = ap_lo, cholesterol = cholesterol, gluc = gluc, smoke = smoke, alco = alco, active = active,  results=results.content.decode('UTF-8'))
  
