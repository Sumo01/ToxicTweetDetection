import pickle
import pandas as pd
from . import retrainer

def predictA(X):
    model=pickle.load(open('model_a.pkl','rb'))
    #mat=retrainer.lstm_process(X)
    prediction=model.predict(X)
    #return prediction
    if prediction==0: return "Not Offensive"
    else: return "Offensive"

def predictB(X):
    notoff=predictA(X)
    if notoff=="Not Offensive":
        return notoff
    #mat=retrainer.lstm_process(X)
    model=pickle.load(open('model_b.pkl','rb'))
    prediction=model.predict(X)
    #return prediction
    if prediction<0.5: return "Directed"
    else: return "Undirected"

def predictC(X):
    undir=predictB(X)
    if undir=="Undirected" or undir=="Not Offensive":
        return undir
    model=pickle.load(open('model_c.pkl','rb'))
    prediction=model.predict(X)
    if prediction==0: return "Group"
    elif prediction==1: return "Individual"
    else: return "Other"

def predict(tweet,type):
    df = retrainer.process(tweet)
    X=df['tweet']
    if type=='A':
        pred= predictA(X)
    elif type=='B':
        pred= predictB(X)
    elif type=='C':
        pred= predictC(X)
    return pred
