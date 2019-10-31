# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:45:15 2019

@author: stany
"""
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics


##Import the spambase dataset and adjust as necessary 
spambase = pd.read_csv('spambase.data',header=None)
spambase.rename(columns={0:"word_freq_make", 1:"word_freq_address", 2:"word_freq_all", 3:"word_freq_3d", 4:"word_freq_our", 
                    5:"word_freq_over", 6:"word_freq_remove", 7:"word_freq_internet", 8:"word_freq_order", 9:"word_freq_mail",
                    10:"word_freq_receive", 11:"word_freq_will", 12:"word_freq_people", 13:"word_freq_report", 14:"word_freq_addresses",
                    15:"word_freq_free", 16:"word_freq_business", 17:"word_freq_email", 18:"word_freq_you", 19:"word_freq_credit", 
                    20:"word_freq_your", 21:"word_freq_font", 22:"word_freq_000", 23:"word_freq_money", 24:"word_freq_hp", 
                    25:"word_freq_hpl", 26:"word_freq_george", 27:"word_freq_650", 28:"word_freq_lab", 29:"word_freq_labs", 
                    30:"word_freq_telnet", 31:"word_freq_857", 32:"word_freq_data", 33:"word_freq_415", 34:"word_freq_85", 
                    35:"word_freq_technology", 36:"word_freq_1999", 37:"word_freq_parts", 38:"word_freq_pm", 39:"word_freq_direct", 
                    40:"word_freq_cs", 41:"word_freq_meeting", 42:"word_freq_original", 43:"word_freq_project", 44:"word_freq_re",
                    45:"word_freq_edu", 46:"word_freq_table", 47:"word_freq_conference", 48:"char_freq_;", 49:"char_freq_(", 
                    50:"char_freq_[", 51:"char_freq_!", 52:"char_freq_$", 53:"char_freq_#", 54:"capital_run_length_average", 
                    55:"capital_run_length_longest", 56:"capital_run_length_total", 57:"is_spam"},inplace=True)
#inplace: Makes changes in original Data Frame if True.


##Split spambase into feature and response sets 
SB_features = spambase.iloc[:, 0:57]
SB_response = spambase.iloc[:, 57]
#SB_response2 = spambase[['is_spam']] this will only select the is_spam column
#spambase.drop(['is_spam'],axis=1)  this will select everything but the is_spam column by dropping 

##Split SB_features and SB_response into training and testing sets (75% and 25% respectively)
SBf_train, SBf_test, SBr_train, SBr_test = train_test_split(
        SB_features, SB_response, test_size=0.25, train_size=0.75, 
        random_state = 0, stratify=SB_response
        )

##Standardize the dataset by first using preprocessing to compute the mean and standard deviation for future scaling
##Then scale the data sets 
SBf_train = preprocessing.StandardScaler().fit_transform(SBf_train.values)
SBf_test = preprocessing.StandardScaler().fit_transform(SBf_test.values)

#Make an instance of the model 
logistic_regression = LogisticRegression() 
lda = LinearDiscriminantAnalysis() 


def kfcv(xtrain,ytrain,kfolds=5,model=None): 
    #Preprocessing 
    xtrain = pd.DataFrame(xtrain) 
    ytrain = pd.DataFrame(ytrain).reset_index() 
    ytrain = ytrain.drop(['index'],axis=1)
    
    #Complete Dataframe 
    concat_df = pd.concat([xtrain,ytrain],ignore_index=True,axis=1)
    
    m,n = np.shape(concat_df)
    
    #Shuffle 
    concat_df.sample(frac=1) 
    
    #Divide training data into K Parts 
    folds = np.array_split(concat_df, kfolds)
    
    validation_errors = [] 
    
    for i in range(kfolds): 
        train_indices = [i for i in range(kfolds)] #creates indices corresponding to each k fold
        del train_indices[i] #delete the index ref kth fold, which will be used for validation  
        train_partitions = [folds[j] for j in train_indices] #copy remaining folds for train to new list 
        training_set = pd.concat(train_partitions,axis=0)
        test_set = pd.DataFrame(folds[i])
        
        train_x = training_set.iloc[:, 0:n-1]
        train_y = training_set.iloc[:,-1]
        test_x = test_set.iloc[:, 0:n-1]
        test_y = test_set.iloc[:,-1]

        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        error = 1 - metrics.accuracy_score(test_y,predictions)
        
        validation_errors.append(error) 
    
    cv_error_metric = np.mean(validation_errors)
    if model == lda: 
        print("Model: Linear Discriminant Analysis")
    elif model == logistic_regression: 
        print("Model: Logistic Regression")
    else: 
        print("Model not recognized.")
    print("Cross Validation Error: ", cv_error_metric)
    print("K Folds: ", kfolds)
    return(None)

kfcv(SBf_train,SBr_train,model=logistic_regression)
kfcv(SBf_train,SBr_train,kfolds=10,model=logistic_regression)
kfcv(SBf_train,SBr_train,model=lda)
kfcv(SBf_train,SBr_train,kfolds=10,model=lda)

