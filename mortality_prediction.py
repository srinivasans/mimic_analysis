'''
Script for Mortality prediction from ICU data 
Author : srinivasan@cs.toronto.edu
'''

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

def normalize_data(data, features):
    for feature in features:
        max_val = data[feature].max()
        min_val = data[feature].min()
        data[feature] = (data[feature] - min_val) / (max_val - min_val)
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
    train_X = normalize_data(train_X, normalize_features)
    train_Y = train_data.loc[:,mortality_target]

    print(train_X.shape)
    print(train_Y.shape)

    test_X = test_data.loc[:,feature_cols]
    test_X = normalize_data(test_X, normalize_features)
    test_Y = test_data.loc[:,mortality_target]

    classifier = LogisticRegression(penalty='l2', C=1.0, random_state=0, solver='lbfgs',multi_class='ovr')
    classifier.fit(train_X, train_Y)
    W = classifier.coef_
    influence_weights = W.argsort(axis=-1)
    
    max_influence = influence_weights[0][0:5]
    min_influence = influence_weights[0][-5:]

    print(max_influence)
    print(np.array(feature_cols)[max_influence])
    print(np.array(feature_cols)[min_influence])

    pred_Y = classifier.predict_proba(test_X)[:,1]
    print(pred_Y.shape)
    auc = roc_auc_score(test_Y, pred_Y)

    print(auc)


if __name__ == '__main__':
    predict_mortality()

