# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:00:36 2019

@author: ASUS
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split 
import re 
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

tqdm.pandas(desc="progress-bar")
stop_words = open("en_stopwords.txt", 'r' , encoding="ISO-8859-1").read()

#Progress bar
def process_data(data):
    data = data.progress_map(preprocess_data)  
    return data

#Preprocessing the data
def preprocess_data(document):

    document = cleaning_data(document)
    document = stopwords_data(document)
    document = stemmer(document)
       
    return document  
    
# Removing the noise and negation handling            
def cleaning_data(review): 
    #Cleaning the data by removing special characters
    review = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", review)
    #Handling negations
    review = re.sub(r"it's", " it is", review)
    review = re.sub(r"ain't", "is not",review)
    review = re.sub(r"aren't", "are not",review)
    review = re.sub(r"couldn't", "could not",review)
    review = re.sub(r"didn't", "did not",review)
    review = re.sub(r"doesn't", "does not",review)
    review = re.sub(r"hadn't", "had not",review)
    review = re.sub(r"hasn't", "has not",review)
    review = re.sub(r"haven't", "have not",review)
    review = re.sub(r"isn't", "is not",review)
    review = re.sub(r"shouldn't", "should not",review)
    review = re.sub(r"shan't", "shall not",review)
    review = re.sub(r"wasn't", "was not",review)
    review = re.sub(r"weren't", "were not",review)
    review = re.sub(r"oughtn't", "ought not",review)
    review = re.sub(r"that's", " that is", review)
    review = re.sub(r"\'s", " 's", review)
    review = re.sub(r"\'ve", " have", review)
    review = re.sub(r"won't", " will not", review)
    review = re.sub(r"wouldn't", " would not", review)
    review = re.sub(r"don't", " do not", review)
    review = re.sub(r"can't", " can not", review)
    review = re.sub(r"cannot", " can not", review)
    review = re.sub(r"n\'t", " n\'t", review)
    review = re.sub(r"\'re", " are", review)
    review = re.sub(r"\'d", " would", review)
    review = re.sub(r"\'ll", " will", review)
    review = re.sub(r"!", " ! ", review)
    review = re.sub(r"\?", " ? ", review)
    review = re.sub(r"\s{2,}", " ", review)
    # Removing all the numbers
    review = re.sub(r'[0-9]+', ' ', review) 
    #Removing all puncs
    review = re.sub(r'[^\w\s]','',review)
    # Substituting multiple spaces with single space
    review = re.sub(r'\s+', ' ', review, flags=re.I)
    #Lower case the data
    review = review.lower()
    return review
    
#Removing the stop words
def stopwords_data(review):
    review = [word for word in review.split() if not word in stop_words]#.words('english')]
    review = ' '.join(review)           
    return review

#list of cleaned words
def cleandata(X_datatrain, X_datatest):
    cleandata_Train = []
    for sen in range(0, len(X_datatrain)): 
        cleandata_Train.append(preprocess_data(str(X_datatrain[sen])))
        
    cleandata_Test = []
    for sen in range(0, len(X_datatest)): 
        cleandata_Test.append(preprocess_data(str(X_datatest[sen])))
    return cleandata_Train, cleandata_Test

def stem(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def stemmer(phrase):
    r=[]
    for word in phrase.split():
        r.append(stem(word))
    return ' '.join(r)

print("Reading of the data files on process...")

df = pd.read_csv("reddit_train.csv")
df.head()
test= pd.read_csv("reddit_test.csv")
X_datatrain = df['comments']
y_datatrain = df['subreddits']

   
X_datatest = test['comments']

print("Task Completed")

print("Cleaning of Training data on process...")
cleandata_Train = process_data(X_datatrain)
print("Task completed")
print("Cleaning of Testing data on process...")
cleandata_Test = process_data(X_datatest)
print("Task completed")

#Count Vectorization:
vectorizer = CountVectorizer(binary=True, analyzer='word', min_df=2,max_df= 0.95, ngram_range=(1, 1))  
X = vectorizer.fit_transform(X_datatrain)

b = X.sum(axis=0).tolist()[0]
# Extract top most occurring words 
# Get the distinct list of words in the comments
a = vectorizer.get_feature_names()
# Zip the frequency of word and the word into a tuple, and make list of tuples 
#and sort by freq descending
freq_freq = [item[1] for item in sorted(list(zip(b,a)), reverse = True, key = lambda x: x[0])]
#choose 20000 most occurring words as features   
vocab =freq_freq[0:20000]
#vectorizing comments binary values based on existence/non-existance of a word 
#belonging to vocabulary features
vectorizer1 = CountVectorizer(binary=True, analyzer='word' , vocabulary = vocab, min_df=2, max_df= 0.95, ngram_range=(1, 1))      

X_train_vect = vectorizer1.fit_transform (cleandata_Train)
       
X_test_vect = vectorizer1.transform(cleandata_Test)
 
#defining a dictionary to convert classes names to integers
dictionary={'anime':0, 'Music':1, 'trees':2, 'conspiracy':3, 'canada':4, 'hockey':5, 
        'worldnews':6, 'funny':7, 'GlobalOffensive':8, 'AskReddit':9, 'nba':10, 
        'nfl':11, 'europe':12, 'soccer':13, 'wow':14 , 'Overwatch':15,
        'gameofthrones':16, 'movies':17, 'leagueoflegends':18,'baseball':19}

y_datatrain = [dictionary[i] for i in y_datatrain] 
#splitting the dataset into test and training set
X_train, X_test, y_train, y_test = train_test_split(X_train_vect, y_datatrain, test_size=0.2, random_state=5)


