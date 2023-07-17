from flask import Flask, render_template, request, redirect, url_for, flash, abort, session, jsonify, Blueprint
import json
import os.path
from werkzeug.utils import secure_filename
from model import SmartPhoneRecommendation
import io
from PIL import Image
import base64

app = Flask(__name__)

@app.route("/")
def home():
    data = io.BytesIO()
    im = Image.open("C:/Users/Savio/fdaFinalReview/static/sm1.jpg")
    im.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    spr = SmartPhoneRecommendation()
    #print(spr.get_top_n())
    return render_template("home.html", li=spr.get_top_n(), img_data=encoded_img_data.decode('utf-8'))
