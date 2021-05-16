from flask import Flask, jsonify, request
from app import mod_dbconn
import keras
import keras.backend as K
import wtte.weibull as weibull
import wtte.wtte as wtte
from keras.models import load_model
import sys
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
from pandas import DataFrame
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import math

app = Flask(__name__)
CORS(app)

#가장 최근 혈당값 10개 전송
@app.route('/bloodsugar')
def bloodsugar():
    db_class = mod_dbconn.Database()

    email = request.args.get("email")
    sql = "SELECT bloodsugar FROM db.bloodsugar order by idx desc limit 10"
    row = db_class.executeAll(sql)

    return jsonify(row)

#add insulin
@app.route('/insulin/add', methods=['POST'])
def addInsulin():

    email = request.form['email']
    insulin = request.form['value']
    fDate = request.form['fDate']

    db_class = mod_dbconn.Database()
    '''
    sql = "INSERT INTO db.insulin (insulin, email, fDate) VALUES(" + insulin + ", \'" + email + "\', \'" + fDate + "\')"
    row = db_class.executeAll(sql)
    print(sql)
    db_class.commit()
    '''
    sql = "SELECT * FROM db.features WHERE email = \'" + email + "\' and fDate >= \'" + fDate + "\' ORDER BY fDate DESC"
    print(sql)
    row = db_class.executeAll(sql)

    #x = 투여량, insulin
    #t = 마지막 투여로부터 지난 시간, fDate - curDate
    for i in row:
        curDate = i.get('fDate')
        x = float(insulin)
        date_time_obj = datetime.strptime(fDate, '%Y-%m-%d %H:%M:%S')
        diff = curDate - date_time_obj
        t = diff.seconds/60
        print(t)
        print("인슐린량")
        print(x)

        input = x*((0.0008*x+0.028)*(0.0008*x + 0.028))*2*t*(2.7**0.09*t)  /  (t*t+(0.0008*x+0.028)*(0.0008*x + 0.028))*(t*t + (0.0008*x + 0.028)*(0.0008*x + 0.028))
        print(insulin.__class__)
        print(curDate.__class__)
        
        sql = "UPDATE db.features SET insulin = "+str(input)+" WHERE fDate = \'"+str(curDate)+"\' and email = \'" + email + "\'"
        print(sql)

        row = db_class.executeAll(sql)
        db_class.commit()

    return jsonify(row)

#delete insulin
@app.route('/insulin/delete')
def deleteInsulin():
    idx = request.args.get("idx")

    db_class = mod_dbconn.Database()

    sql = "DELETE FROM db.insulin WHERE idx="+idx
    row = db_class.executeAll(sql)
    db_class.commit()

    return jsonify(row)

#read insulin
@app.route('/insulin/read')
def readInsulin():

    email = request.args.get("email")
    
    db_class = mod_dbconn.Database()

    sql = "SELECT * FROM db.insulin WHERE email="+email
    row = db_class.executeAll(sql)

    return jsonify(row)

#add carb
@app.route('/carb/add', methods=['POST'])
def addCarb():

    email = request.form['email']
    carb = request.form['value']
    fDate = request.form['fDate']

    db_class = mod_dbconn.Database()

    '''
    sql = "INSERT INTO db.carb (carb, email, fDate) VALUES(" + carb + ", \'" + email + "\', \'" + fDate + "\')"
    row = db_class.executeAll(sql)
    print(sql)
    db_class.commit()
    '''
    sql = "SELECT * FROM db.features WHERE email = \'" + email + "\' and fDate >= \'" + fDate + "\' ORDER BY fDate DESC"
    print(sql)
    row = db_class.executeAll(sql)

    #x = 투여량, insulin
    #t = 마지막 투여로부터 지난 시간, fDate - curDate
    for i in row:
        curDate = i.get('fDate')
        v = float(carb)
        date_time_obj = datetime.strptime(fDate, '%Y-%m-%d %H:%M:%S')
        diff = curDate - date_time_obj
        t = diff.seconds/60
        print(t)

        if t>= 0 and t < 30:
            input = (t-1)/15 + math.exp(-t)
        elif t >= 30 and t < 30 + (5.5*v-2) / 240:
            input = 2+math.exp(-t)
        elif 30 + (5.5*v-2) / 240 <= t and t < 60 + (5.5*v-2) / 240:
            input = -0.0015*v + math.exp(-t) -0.06*t +4.06
        else :
            input = math.exp(-t)
        
        sql = "UPDATE db.features SET carb = "+str(input)+" WHERE fDate = \'"+str(curDate)+"\' and email = \'"+email+"\'"
        print(sql)

        row = db_class.executeAll(sql)
        db_class.commit()

    return jsonify(row)

#delete carb
@app.route('/carb/delete')
def deleteCarb():
    idx = request.args.get("idx")

    db_class = mod_dbconn.Database()

    sql = "DELETE FROM db.carb WHERE idx="+idx
    row = db_class.executeAll(sql)
    db_class.commit()

    return jsonify(row)

#read carb
@app.route('/carb/read')
def readCarb():
    db_class = mod_dbconn.Database()

    email = request.args.get("email")

    sql = "SELECT * FROM db.carb WHERE email="+email
    row = db_class.executeAll(sql)

    return jsonify(row)

#get model
MODEL_PATH = './app/wtte_reuse_model_3.h5'

def weibull_loglik_discrete(y_true, y_pred, epsilon=K.epsilon()):
    y = y_true[..., 0]
    u = y_true[..., 1]
    a = y_pred[..., 0]
    b = y_pred[..., 1]
    
    hazard0 = K.pow((y + epsilon) / a, b)
    hazard1 = K.pow((y + 1.0) / a, b)

    loss = u * K.log(K.exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1
    return -loss

def activation_weibull(y_true):
    a = y_true[..., 0]
    b = y_true[..., 1]
        
    a = K.exp(a)
    b = K.sigmoid(b)
    return K.stack([a, b], axis=-1)

# It can be used to reconstruct the model identically.
#reconstructed_model = keras.models.load_model("./wtte_reuse_model_3.h5")

model = load_model("./wtte_reuse_model_3.h5",
                    custom_objects = {"./weibull_loglik_discrete": weibull_loglik_discrete,"activate":activation_weibull, 
                                        "loss_function":wtte.Loss(kind='discrete', clip_prob=1e-5).loss_function})

def predict_model(input, model):
    return model.predict(input)

def pad_sequence(df, max_seq_length, mask=0):
    """
    Applies right padding to a sequences until max_seq_length with mask 
    """
    return np.pad(df.values, ((0, max_seq_length - df.values.shape[0]), (0,0)), 
                  "constant", constant_values=mask)
  
def pad_engines(df, cols, max_batch_len, mask=0):
    """
    Applies right padding to the columns "cols" of all the engines 
    """
    return np.array([pad_sequence(df[df['id'] == batch_id][cols], max_batch_len, mask=mask) 
                     for batch_id in df['id'].unique()])

from sklearn import preprocessing

#return machine learning result
@app.route('/tte', methods=['POST'])
def user():
    email = request.form['email']
    #email = request.args.get("email")

    print(email)

    db_class = mod_dbconn.Database()

    sql = "SELECT id, cycle, bloodsugar, carb, insulin FROM db.features WHERE email=\""+email+"\"ORDER BY fDate DESC LIMIT 1301"
    row = db_class.executeAll(sql)

    train_df = DataFrame(row)

    print(train_df)
    print(train_df.shape)

    train_df = train_df.sort_values(['id','cycle'])

    # Data Labeling - generate column RUL (Remaining Useful Life or Time to Failure)
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)

    print("러벨링")
    print(train_df.shape)

    # MinMax normalization (from 0 to 1)
    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2', 'label3'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=train_df.index)
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns = train_df.columns)

    print("노말라이제이션")
    print(train_df.shape)

    train_df[train_df["id"] == 1].tail()

    max_batch_len = train_df['id'].value_counts().max()
    train_cols = ['bloodsugar', 'carb', 'insulin'] + ['cycle_norm']
    test_cols = ["RUL"]

    xhat = pad_engines(train_df, train_cols, 1301)

    print(xhat)
    print(xhat.shape)

    #(1301, 4)
    print(xhat.shape)

    #xhat = np.zeros((1, 1301, 4))
    y_pred = model.predict(xhat)

    print(y_pred.shape)
    print(y_pred[-1].shape)

    #tte = {
    #'tte': json.dumps(str(y_pred[-1].flatten()[-1]))
    #'tte':str(y_pred[-1].flatten()[-1])
    #}

    return str(y_pred[-1].flatten()[-1])

@app.route('/tte2', methods=['POST', 'GET'])
def hi():
    email = request.form['email']

    print(email)

    xhat = np.zeros((3, 1301, 4))
    y_pred = model.predict(xhat)
    print(y_pred[-1].flatten()[-1])

    tte = {
    'tte': json.dumps(str(y_pred[-1].flatten()[-1]))
    }
    return jsonify(tte)

if __name__=='__main__':
    app.run(debug=True)

from app import app