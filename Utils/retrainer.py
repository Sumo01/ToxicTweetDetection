import pandas as pd
import nltk 
from nltk import word_tokenize
from nltk.corpus import stopwords as st
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

is_running_status = False

def remove_stopwords(df):
    stopwords = set(st.words("english"))
    df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords]))
    df['tweet'] = df['tweet'].str.strip()  # Trim leading and trailing whitespaces
    return df

def clean_tweets(df):
    stopwords = set(st.words("english"))
    punctuations = string.punctuation
    
    df.loc[:, 'tweet'] = df.tweet.str.replace('@USER', '') #Remove mentions (@USER)
    df.loc[:, 'tweet'] = df.tweet.str.replace('URL', '') #Remove URLs
    df.loc[:, 'tweet'] = df.tweet.str.replace('&amp', 'and') #Replace ampersand (&) with and
    df.loc[:, 'tweet'] = df.tweet.str.replace('&lt','') #Remove &lt
    df.loc[:, 'tweet'] = df.tweet.str.replace('&gt','') #Remove &gt
    df.loc[:, 'tweet'] = df.tweet.str.replace('\d+','') #Remove numbers
    df.loc[:, 'tweet'] = df.tweet.str.lower() #Lowercase

    #Remove punctuations
    for punctuation in punctuations:
        df.loc[:, 'tweet'] = df.tweet.str.replace(punctuation, '')

    df.loc[:, 'tweet'] = df.astype(str).apply(
        lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')
    )
    #Remove emojis
    df.loc[:, 'tweet'] = df.tweet.str.strip() #Trim leading and trailing whitespaces
    df=remove_stopwords(df)
    return df
    
def lstm_process(X):
    tok=Tokenizer(1000)
    tok.fit_on_texts(X)
    seq=tok.texts_to_sequences(X)
    seq_matrix=sequence.pad_sequences(seq,150)
    return seq_matrix


def process(tweet):
    df=pd.DataFrame({"tweet": tweet}, index=[0])
    df.update(df[['tweet']].applymap('\'{}\''.format))
    clean_df=clean_tweets(df)
    return clean_df

