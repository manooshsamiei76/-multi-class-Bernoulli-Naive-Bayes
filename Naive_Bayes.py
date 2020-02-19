# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:10:17 2019

@author: ASUS
"""

import data_preprocessing as dp
import numpy as np
import scipy.sparse

class NaiveBayes: 
    
    def __init__(self):
        
        self.theta_k=0.
        self.theta_jk=0.
        
    def fit(self,X,y,features):
        
        
        index=[]
        self.theta_jk=np.zeros((20,len(features)))
        self.theta_k=np.zeros(20)
        #calculating theta  for each 
        for k in range(20):
            #finding indexes where inputs are in class k
            index=np.where(y==k)
            #finding pobability of each class
            self.theta_k[k] = (len(index[0]))/float(y.shape[0])
            #computing conditional probability of each feature for each class 
            #Laplace smoothing is used to deal with words which are not observed 
            #in the training data but are available in the training set
            self.theta_jk[k][:] = (scipy.sparse.csr_matrix.sum(X[index],axis=0)+1)/(float(len(index[0]))+2)
                  
        return
    
    def predict(self,X):
        
        prob_y=np.zeros((20,len(X)))
        
        for k in range(20):
            
            w0 = np.log(self.theta_k[k])+((np.log(1-self.theta_jk[k][:])).sum())
            w = np.log(self.theta_jk[k][:])-np.log(1-self.theta_jk[k][:])
            #The probabilitis of being in each of k classes for each of input data(comments)
            prob_y[k]=np.matmul(X,w.T) + w0  
        #selecting the class with higher probability for each input data (comment)
        y_predict=np.argmax(prob_y.T, axis=1)

        return y_predict
    
    def evaluate_acc(self,y,y_pred):
        #number of predictions       
        num_pred = np.float(len(y))
        #calculating average accuracy over all classes
        average_acc = np.sum((y==y_pred)*1)/num_pred
        #calculating accuracy for each class 
        acc=np.zeros(20)
        for k in range(20): 
            index=np.where(y==k)
            num_pred=np.float(len(index[0]))
            #acc array contains accuracies for each of 20 classes in the order of
            #the below dictionary:
            #dictionary={'anime':0, 'Music':1, 'trees':2, 'conspiracy':3, 'canada':4, 'hockey':5, 
            #'worldnews':6, 'funny':7, 'GlobalOffensive':8, 'AskReddit':9, 'nba':10, 
            #'nfl':11, 'europe':12, 'soccer':13, 'wow':14 , 'Overwatch':15,
            #'gameofthrones':16, 'movies':17, 'leagueoflegends':18,'baseball':19} 
            acc[k]= (np.sum(y_pred[index]==k).astype(np.int))/num_pred
        return acc,average_acc
    
def test():
    
    
    d = NaiveBayes()
    
    feature_labels = dp.vocab
    
    X_training = dp.X_train  
    y_training = np.array(dp.y_train)
    X_testing = dp.X_test.toarray()
    y_testing = np.array(dp.y_test)
    
    
    d.fit(X_training, y_training, feature_labels)
    y_prediction = d.predict(X_testing)

    (acc, average_acc) = d.evaluate_acc(y_testing, y_prediction)
    print("Average accuracy on test set: ", average_acc)
    print("Accuracy on test set for each class (in the order of the dictionary): ", acc)

if __name__ == "__main__":        
    test()     
        
