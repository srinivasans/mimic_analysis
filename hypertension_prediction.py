'''
Script for predicting Hypertension from MIMIC data 
Author : srinivasan@cs.toronto.edu
'''
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def fit_model(train_data, columns):
    train_X = train_data[columns]
    train_Y = train_data['hypertension']
    model = LogisticRegression(penalty='l2', C=3.0, random_state=0, solver='lbfgs', multi_class='ovr', max_iter=10000)
    model.fit(train_X, train_Y)
    return model

def test_model(model, test_data, columns):
    test_X = test_data[columns]
    test_Y = test_data['hypertension']

    pred_Y = model.predict_proba(test_X)[:,1]
    
    fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
    plt.plot(fpr, tpr)
    plt.show()
    auc = roc_auc_score(test_Y, pred_Y)
    print(auc)

def predict_hypertension():
    mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
    patients = pd.read_csv(os.path.join(mimicdir, 'hypertension_patients.gz'), compression='gzip')
    charts = pd.read_csv(os.path.join(mimicdir, 'hypertension_charts.gz'), compression='gzip')
    print(patients.loc[:,'hadm_id'])
    print(list(charts))
    print(list(patients))

    means = pd.DataFrame(charts.groupby(['hadm_id','itemid'],axis=0)['valuenum'].mean())
    maxes = pd.DataFrame(charts.groupby(['hadm_id','itemid'],axis=0)['valuenum'].max())
    mins = pd.DataFrame(charts.groupby(['hadm_id','itemid'],axis=0)['valuenum'].min())

    means = means.reset_index().pivot('hadm_id', 'itemid', 'valuenum')
    maxes = maxes.reset_index().pivot('hadm_id', 'itemid', 'valuenum')
    mins = mins.reset_index().pivot('hadm_id', 'itemid', 'valuenum')

    feature_map = {220045: 'heartrate', 220210: 'rbpm', 220277:'spo2', 220181:'bpmm'}

    means.rename(columns=feature_map, inplace=True)
    means = means.add_suffix('_mean')
    maxes.rename(columns=feature_map, inplace=True)
    maxes = maxes.add_suffix('_max')
    mins.rename(columns=feature_map, inplace=True)
    mins= mins.add_suffix('_min')

    data = pd.concat([mins, maxes, means],axis=1)
    data = data.merge(patients, how='left', left_on='hadm_id', right_on='hadm_id')
    data.dropna(axis=0, inplace=True)

    hr_columns = ['heartrate_min', 'heartrate_max', 'heartrate_mean']
    rbpm_columns = ['rbpm_min', 'rbpm_max', 'rbpm_mean']
    spo2_columns = ['spo2_min', 'spo2_max', 'spo2_mean']
    bpmm_columns = ['bpmm_min', 'bpmm_max', 'bpmm_mean']

    train_data = data[data['train']==1]
    test_data = data[data['train']==0]

    # Evaluate Heart Rate model
    model = fit_model(train_data, hr_columns)
    test_model(model, test_data, hr_columns)

    # Evaluate Respiratory BPM model
    model = fit_model(train_data, rbpm_columns)
    test_model(model, test_data, rbpm_columns)

    # Evaluate SpO2 Model
    model = fit_model(train_data, spo2_columns)
    test_model(model, test_data, spo2_columns)

    # Evaluate BPMM Model
    model = fit_model(train_data, bpmm_columns)
    test_model(model, test_data, bpmm_columns)

    # Evaluate using all the features
    model = fit_model(train_data, hr_columns+rbpm_columns+spo2_columns+bpmm_columns)
    test_model(model, test_data, hr_columns+rbpm_columns+spo2_columns+bpmm_columns)


if __name__ == '__main__':
    predict_hypertension()