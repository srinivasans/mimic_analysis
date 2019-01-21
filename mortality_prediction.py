'''
Script for Mortality prediction from ICU data 
Author : srinivasan@cs.toronto.edu
'''

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

feature_max = {}
feature_min = {}
feature_mean = {}
feature_std = {}

def min_max_normalize(feature, feature_name, test=False):
    if not test:
        max_val = feature.max()
        min_val = feature.min()
        feature_max[feature_name] = max_val
        feature_min[feature_name] = min_val
    else:
        max_val = feature_max[feature_name]
        min_val = feature_min[feature_name]

    feature = (feature - min_val) / (max_val - min_val)
    return feature


def mean_std_normalize(feature, feature_name, test=False):
    if not test:
        mean = feature.mean()
        std = feature.std()
        feature_mean[feature_name] = mean
        feature_std[feature_name] = std
    else:
        mean = feature_mean[feature_name]
        std = feature_std[feature_name]

    feature = (feature - mean) / (std)
    return feature

def normalize_data(data, features, type='min_max', test=False):
    for feature in features:
        if type=='min_max':
            data[feature] = min_max_normalize(data[feature], feature, test)
        elif type=='mean_std':
            data[feature] = mean_std_normalize(data[feature], feature, test)
    return data

def predict_mortality():
    mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
    data = pd.read_csv(os.path.join(mimicdir, 'adult_icu.gz'), compression='gzip')
    print(list(data))

    train_data = data.loc[data['train'] == 1]
    test_data = data.loc[data['train'] == 0]
    
    feature_cols = ['age', 'first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white', 'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN', 'admType_URGENT',
     'heartrate_min', 'heartrate_max', 'heartrate_mean', 'sysbp_min', 
     'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 
     'meanbp_min', 'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max', 
     'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 
     'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin', 
     'bicarbonate', 'bilirubin', 'creatinine','chloride', 'glucose', 'hematocrit', 
     'hemoglobin', 'lactate', 'magnesium', 'phosphate', 'platelet', 'potassium', 'ptt', 
     'inr', 'pt', 'sodium', 'bun', 'wbc']

    normalize_features = ['age', 'heartrate_min','heartrate_max','heartrate_mean', 'sysbp_min', 
     'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 
     'meanbp_min', 'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max', 
     'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 
     'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean','aniongap', 'albumin', 
     'bicarbonate', 'bilirubin', 'creatinine','chloride', 'glucose', 'hematocrit', 
     'hemoglobin', 'lactate', 'magnesium', 'phosphate', 'platelet', 'potassium','ptt', 
     'inr', 'pt', 'sodium', 'bun', 'wbc']
    mortality_target = ['mort_icu']

    train_X = train_data.loc[:,feature_cols]
    train_X = normalize_data(train_X, normalize_features,type='min_max', test=False)
    train_Y = train_data.loc[:,mortality_target]

    print(train_X.shape)
    print(train_Y.shape)

    test_X = test_data.loc[:,feature_cols]
    test_X = normalize_data(test_X, normalize_features,type='min_max', test=True)
    test_Y = test_data.loc[:,mortality_target]

    classifier = LogisticRegression(penalty='l2', C=1.0, random_state=0, solver='lbfgs', multi_class='ovr')
    classifier.fit(train_X, train_Y)
    W = classifier.coef_
    influence_weights = W.argsort(axis=-1) # Not absolute value as negative coeffs affect the model but reduce the risk factor 
    
    min_influence = influence_weights[0][0:5]
    max_influence = influence_weights[0][-5:][::-1]

    print(max_influence)
    print(np.array(feature_cols)[max_influence])
    print(np.array(feature_cols)[min_influence])


    pred_Y = classifier.predict_proba(test_X)[:,1]
    print(pred_Y.shape)

    fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
    plt.plot(fpr, tpr)
    plt.show()

    auc = roc_auc_score(test_Y, pred_Y)

    print(auc)

def predict_mortality_from_notes():
    mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
    data = pd.read_csv(os.path.join(mimicdir, 'adult_notes.gz'), compression='gzip')
    print(list(data))

    train_data = data.loc[data['train'] == 1]
    test_data = data.loc[data['train'] == 0]
    
    print(train_data.shape)
    
    for i in range(0, train_data.shape[0]):
        text = train_data.iloc[i]['chartext']
        try:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
        except:
            tokens = ['']
        stop = set(stopwords.words('english'))
        clean_tokens = [k for k in tokens if k not in stop]
        #print(clean_tokens)
        train_data.iloc[i,train_data.columns.get_loc('chartext')] = (' ').join(clean_tokens)
        print(i)

    print(train_data.shape)
    vectorizer = TfidfVectorizer()
    train_X = vectorizer.fit_transform(train_data.loc[:,'chartext'].values.astype('U'))
    train_Y = train_data.loc[:,'mort_icu']

    print(train_data.shape)
    print(train_X.shape)

    classifier = LogisticRegression(penalty='l2', C=1.0, random_state=0, solver='lbfgs', multi_class='ovr')
    classifier.fit(train_X, train_Y)
    W = classifier.coef_
    influence_weights = W.argsort(axis=-1)

    for i in range(0, test_data.shape[0]):
        text = test_data.iloc[i]['chartext']
        try:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
        except:
            tokens = ['']
        stop = set(stopwords.words('english'))
        clean_tokens = [k for k in tokens if k not in stop]
        test_data.iloc[i,test_data.columns.get_loc('chartext')] = (' ').join(clean_tokens)

    test_X = vectorizer.transform(test_data.loc[:,'chartext'].values.astype('U'))
    test_Y = test_data.loc[:,'mort_icu']

    pred_Y = classifier.predict_proba(test_X)[:,1]
    print(pred_Y.shape)
    auc = roc_auc_score(test_Y, pred_Y)
    
    print(auc)
    

if __name__ == '__main__':
    #predict_mortality()
    predict_mortality_from_notes()
