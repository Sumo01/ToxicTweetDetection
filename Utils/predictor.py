import pickle
import pandas as pd
from . import retrainer

def predict(tweet):
    df = retrainer.process(tweet)
    X=df['tweet']
    # Load the model
    model = pickle.load(open('taskamnb.pkl', 'rb'))
    # Make prediction
    prediction = model.predict(X)
    # Return prediction
    return prediction
