from app import app
from flask import Blueprint
import keras
import keras.backend as K
import wtte.weibull as weibull
import wtte.wtte as wtte
from keras.models import load_model
import sys
from flask_ngrok import run_with_ngrok
import os

if __name__ == "__main__":
    #run_with_ngrok(app)
    app.run(host='0.0.0.0') # 127.0.0.1